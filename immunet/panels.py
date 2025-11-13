from enum import Enum
import json
from pathlib import Path
from typing import Callable, List, Optional, Sequence
import warnings
import numpy as np

from config import PANEL_FILE

VALID_MARKERS = [
    "CD3",
    "FOXP3",
    "CD20",
    "CD56",
    "CD8",
    "CD45RO",
    "TM",
]

VALID_MAINTYPES = [
    "T cell",
    "B cell",
    "NK cell",
    "NKT cell",
    "Tumor cell",
    "Other cell",
    "No cell",
    "Neural structure",
    "Invalid",
]


VALID_SUBTYPES = [
    # T Types
    "Cytotoxic T cell",
    "Regulatory T cell",
    "Helper T cell",
    "Memory T cell",
    "CD3+ T cell",
    "CD8+ T cell",
    "CD45RO+ T cell",
    "FOXP3+ T cell",
    "Other T cell",
    "Invalid T cell",
]


class Phenotype(Enum):
    NO = "No cell"
    OTHER = "Other cell"
    INVALID = "Invalid"


class MarkerExpression(Enum):
    POSITIVE = True
    NEGATIVE = False
    WILDCARD = "*"

    @classmethod
    def from_likert(
        cls, likert_value: int, likert_cutoff: int = 3
    ) -> "MarkerExpression":
        if likert_value > likert_cutoff:
            return MarkerExpression.POSITIVE
        else:
            return MarkerExpression.NEGATIVE

    @classmethod
    def from_string(cls, _str: str) -> "MarkerExpression":
        if _str.lower() == "true":
            return MarkerExpression.POSITIVE
        if _str.lower() == "false":
            return MarkerExpression.NEGATIVE
        else:
            return MarkerExpression.WILDCARD

    @classmethod
    def from_prediction(
        cls, prediction_value: float, threshold: float
    ) -> "MarkerExpression":
        if prediction_value >= threshold:
            return MarkerExpression.POSITIVE
        else:
            return MarkerExpression.NEGATIVE

    def matches(self, other: "MarkerExpression") -> bool:
        """Wildcard-aware match: returns True if either side is wildcard or values equal."""
        if not isinstance(other, MarkerExpression):
            return False
        if self.is_wildcard() or other.is_wildcard():
            return True
        return self.value == other.value

    def is_wildcard(self) -> bool:
        return self.value == MarkerExpression.WILDCARD.value


class Cell:
    """
    Represents a cell, with possible empty subtype.
    """

    def __init__(
        self,
        maintype: str,
        subtype: str = "",
        markers: dict[str, MarkerExpression] = {},
        background: bool = False,
    ):
        self.maintype = maintype
        self.subtype = subtype
        self.markers = markers
        self.background = background

        if len(markers.keys()) != 0:
            if self.background:
                warnings.warn(
                    f"Markers provided to background cell {self}, they will be ignored!"
                )
                self.markers = {}

    @property
    def name(self):
        return self.subtype if self.subtype != "" else self.maintype
    
    @property
    def main_type_name(self):
        warnings.warn(f'The use of `main_type_name` is deprecated. Use maintype instead', DeprecationWarning)
        return self.maintype

    @property
    def is_invalid(self):
        return self.maintype == Phenotype.INVALID.value
    
    @property
    def is_inconsistent(self):
        warnings.warn(f'The use of `inconsistent` is deprecated.', DeprecationWarning)
        return self.is_invalid

    @property
    def other(self):
        return self.maintype == Phenotype.OTHER.value

    @classmethod
    def from_likert(
        cls,
        maintype: str,
        subtype: str,
        likert_array: list[int],
        panel_markers: list[str],
    ) -> "Cell":
        markers = {}
        for i, likert_value in enumerate(likert_array):
            expression = MarkerExpression.from_likert(likert_value)
            markers[panel_markers[i].upper()] = expression
        return cls(maintype, subtype, markers)

    @classmethod
    def from_dict(cls, dictionary: dict):
        markers = {}
        for marker, pheno in dictionary["phenotype"].items():
            markers[marker.upper()] = MarkerExpression.from_string(str(pheno))

        return cls(
            dictionary["type"],
            "" if "subtype" not in dictionary else dictionary["subtype"],
            markers,
        )

    def __str__(self):
        representation = self.maintype
        if self.subtype:
            representation += f" ({self.subtype})"
        return representation


class Panel:
    """Representation of a antibody panel."""

    def __init__(
        self,
        name: str,
        markers: List[str],
        decorable_cells: List[Cell],
        undecorable_cells: List[Cell],
        background_cells: List[Cell],
        strict: bool = True,
    ):
        self.name = name
        self.markers = markers
        self.__decorable_cells: List[Cell] = decorable_cells
        self.__undecorable_cells: List[Cell] = undecorable_cells
        self.__foreground_cells: List[Cell] = decorable_cells + undecorable_cells
        self.__background_cells: List[Cell] = background_cells
        self.__cells: List[Cell] = self.__foreground_cells + self.__background_cells
        self.strict = strict  # if false, then the cell generated will be based on its markers, so invalid cells are never generated

    @property
    def fg_cell_types(self):
        return list(set([cell.maintype for cell in self.__foreground_cells]))

    @property
    def cell_types(self) -> list[str]:
        """List of all cell main types."""
        return list(set([cell.maintype for cell in self.__cells]))

    @property
    def decorable_types(self) -> list[str]:
        """List of all decorable main types."""
        return list(set([cell.maintype for cell in self.__decorable_cells]))

    @classmethod
    def from_dict(cls, dictionary: dict):
        """Builds a panel from a dictionary, must have valid format."""
        required_keys = ["panel", "markers", "phenotypes"]

        for req_key in required_keys:
            if req_key not in dictionary:
                raise ValueError(f"Panel dictionary has to contain {req_key} key")

        panel_name = dictionary["panel"]
        if not isinstance(panel_name, str):
            raise ValueError("Panel property has to be a string")

        panel_markers = [x.upper() for x in dictionary["markers"]]
        for p_marker in panel_markers:
            if p_marker not in VALID_MARKERS:
                warnings.warn(
                    f"[{panel_name}] Unknown or invalid marker! {p_marker} in panel {panel_name}. Please check spelling!"
                )

        if (
            not isinstance(panel_markers, list)
            or len(panel_markers) == 0
            or not all(isinstance(pm, str) for pm in panel_markers)
        ):
            raise ValueError("Markers property has to be a list of string values")

        # Panels are strict by default
        strict = True
        if "strict" in dictionary:
            strict = dictionary["strict"]

        background_cells = []
        foreground_cells_decorated = []
        foreground_undecorated_cells = []
        for cell_dict in dictionary["phenotypes"]:
            if cell_dict["type"] not in VALID_MAINTYPES:
                warnings.warn(
                    f"[{panel_name}] Unknown or invalid cell maintype type! {cell_dict['type']} in panel {panel_name}. Please check spelling!"
                )
            if "background" in cell_dict:
                background_cells.append(Cell(cell_dict["type"], background=True))
            else:
                cell = Cell.from_dict(cell_dict)
                if "subtype" in cell_dict and cell_dict["subtype"] != "":
                    if cell_dict["subtype"] not in VALID_SUBTYPES:
                        warnings.warn(
                            f"[{panel_name}] Unknown or invalid cell sub type! {cell_dict['subtype']} in panel {panel_name}. Please check spelling!"
                        )
                    foreground_cells_decorated.append(cell)
                else:
                    foreground_undecorated_cells.append(cell)

        panel = cls(
            name=panel_name,
            markers=panel_markers,
            decorable_cells=foreground_cells_decorated,
            undecorable_cells=foreground_undecorated_cells,
            background_cells=background_cells,
            strict=strict,
        )
        return panel

    def cell_from_annotation(
        self, _type: str, likert_array: Optional[list[int]]
    ) -> Cell:
        """Create cell object based on an annotation.

        Parameters
        ----------
        _type : str
            Cell main type of the annotation
        likert_array : Optional[list[int]]
            Positivity list in likert scale ([0-1]), leave None if cell is not decorable (does not have a subtype)

        Returns
        -------
        Cell
            Created cell inferred from main type and positivity.
        """

        if _type not in self.cell_types and self.strict:
            return Cell(Phenotype.INVALID.value)
        elif _type not in self.decorable_types:
            # undecorable cell
            return Cell(_type, subtype="")
        else:
            if not likert_array:
                raise ValueError(
                    f"Passed empty likert scale but {_type} is decorable, so it must be provided!"
                )
            return self.decorable_cell_from_annotation(_type, likert_array)

    def decorable_cell_from_annotation(
        self, _type: str, likert_array: list[int], likert_cutoff=3
    ) -> Cell:
        """Creates decorable cell from annotation.

        Parameters
        ----------
        _type : str
            Cell main type
        likert_array : list[int]
            Positivity list in likert scale ([0-1]), leave None if cell is not decorable (does not have a subtype)

        Returns
        -------
        Cell
            Decorated cell inferred from main type and positivity.

        """
        if _type not in self.decorable_types and self.strict:
            raise ValueError(f"Only {self.decorable_types} decorable in this panel")

        return self.__build_cells(
            likert_array, MarkerExpression.from_likert, likert_cutoff
        )

    def cell_from_prediction(
        self,
        distance: float,
        phenotype: list[float],
        activ_th=0.4,
        detection_th=3.5,
        mm_pp=2,
    ) -> Cell:
        """Creates decorable cell from annotation.

        Parameters
        ----------
        distance : float
            Distance (in pixels) to cell center.
        phenotype : list[float]
            Predicted phenotype list.
        activ_th : float, optional
            Threshold for consider phenotype active, by default 0.4
        detection_th : float, optional
            Distance threshold (in pixels) for a cell to be considered, by default 3.5
        mm_pp : int, optional
            Conversion from pixels to milimeter, by default 2

        Returns
        -------
        Cell
            Predicted cell.
        """
        if distance > detection_th * mm_pp:
            return Cell(Phenotype.NO.value)

        # No expression is predicted
        if sum(np.array(phenotype) > activ_th) == 0:
            return Cell(Phenotype.OTHER.value)

        return self.__build_cells(phenotype, MarkerExpression.from_prediction, activ_th)

    def __literal_cell_from_expression(
        self,
        expressions: Sequence[int | float],
        cell_method: Callable[..., "MarkerExpression"],
        threshold: int | float,
    ) -> Cell:
        """Internal method that generates a custom cell name based on the active markers.
        e.g., [CD3+CD20-FOXP3- cell]
        """
        custom_cell_name = []
        cell_markers = {}
        for i, expression in enumerate(expressions):
            marker = self.markers[i]
            ann_expression = cell_method(expression, threshold)
            ann_expression_sign = (
                "+" if ann_expression == MarkerExpression.POSITIVE else "-"
            )
            custom_cell_name.append(f"{marker}{ann_expression_sign}")
            cell_markers[marker] = ann_expression

        return Cell(" ".join(custom_cell_name), "", cell_markers)

    def __build_cells(
        self,
        expressions: Sequence[int | float],
        cell_method: Callable[..., "MarkerExpression"],
        threshold: int | float,
    ) -> Cell:
        """Internal method for handling logic of cell object creation."""
        if not self.strict:
            # override when using a non-strict panel
            return self.__literal_cell_from_expression(
                expressions, cell_method, threshold
            )

        possible_matches: List[Cell] = []
        for cell in self.__foreground_cells:
            for i, expr in enumerate(expressions):
                marker = self.markers[i]
                ann_expression = cell_method(expr, threshold)
                cell_expression = cell.markers.get(marker)
                if not cell_expression:
                    raise ValueError(
                        f"Cell {cell} is missing marker {marker}! Please revise the panel definition."
                    )
                if not cell_expression.matches(ann_expression):
                    break
                if i == (len(expressions) - 1):
                    possible_matches.append(cell)

        if len(possible_matches) == 1:
            return possible_matches[0]
        elif len(possible_matches) > 1:
            # find the most specific cell (less wildcards)
            best_match: tuple[Cell | None, int] = (None, 99999999999)
            for cell in possible_matches:
                num_of_wildcard = sum(x.is_wildcard() for x in cell.markers.values())
                if num_of_wildcard < best_match[1]:
                    best_match = (cell, num_of_wildcard)
            if best_match[0]:
                return best_match[0]
        
        return Cell(Phenotype.INVALID.value)


with open(Path(PANEL_FILE)) as f:
    panels_info = json.load(f)

panels: dict[str, Panel] = {}
for panel_info in panels_info:
    panel = Panel.from_dict(panel_info)
    panels[panel.name] = panel
