"""
Pygame UI Code Generator Engine
================================
Reads a structured JSON layout file (produced by the OpenCV sketch-recognition
pipeline) and emits a clean, self-contained, OOP Pygame Python module.

JSON Format
-----------
{
  "ui_name": "GameHUD",          # becomes the Python class name
  "screen_width": 800,
  "screen_height": 600,
  "background_color": [30, 30, 30],
  "elements": [
    {
      "id":        "health_bar_1",   # unique Python identifier
      "type":      "health_bar",     # see SUPPORTED_TYPES below
      "x": 20, "y": 20, "w": 200, "h": 30,
      "label":     "HP",             # optional
      "color":     [200, 0, 0],      # [R,G,B] or named string
      "value":     75,               # health_bar / progress_bar: current
      "max_value": 100               # health_bar / progress_bar: maximum
    },
    ...
  ]
}

Supported element types
-----------------------
  button       – clickable rectangle with label, hover & click state
  health_bar   – coloured fill bar with background, current/max values
  progress_bar – generic progress bar (identical behaviour to health_bar)
  text_label   – static or dynamic text drawn on the surface
  panel        – background rectangle / container
  icon         – placeholder rectangle with a centred label

Usage (CLI)
-----------
  python code_generator.py --input layout.json --output generated_ui.py
  python code_generator.py --input layout.json --output ui.py --class-name MyUI

Usage (library)
---------------
  from code_generator import generate_pygame_ui
  generate_pygame_ui("layout.json", "generated_ui.py", class_name="GameHUD")

Author : Ákos (Person 2 – Code Generator Engine)
Project: Mesterséges Intelligencia Beadandó
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import textwrap
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("CodeGeneratorEngine")

# ---------------------------------------------------------------------------
# Constants & type aliases
# ---------------------------------------------------------------------------

JsonDict = Dict[str, Any]

SUPPORTED_TYPES: Tuple[str, ...] = (
    "button",
    "health_bar",
    "progress_bar",
    "text_label",
    "panel",
    "icon",
)

REQUIRED_ELEMENT_FIELDS: Tuple[str, ...] = ("id", "type", "x", "y", "w", "h")

# Named colour → RGB mapping (superset of CSS basic colours)
NAMED_COLORS: Dict[str, Tuple[int, int, int]] = {
    "red":       (220,  20,  60),
    "green":     ( 34, 139,  34),
    "blue":      ( 70, 130, 180),
    "yellow":    (255, 215,   0),
    "orange":    (255, 140,   0),
    "purple":    (128,   0, 128),
    "pink":      (255, 105, 180),
    "white":     (255, 255, 255),
    "black":     (  0,   0,   0),
    "grey":      (128, 128, 128),
    "gray":      (128, 128, 128),
    "cyan":      (  0, 255, 255),
    "magenta":   (255,   0, 255),
    "brown":     (139,  69,  19),
    "gold":      (255, 215,   0),
    "silver":    (192, 192, 192),
    "darkred":   (139,   0,   0),
    "darkgreen":  ( 0, 100,   0),
    "darkblue":  (  0,   0, 139),
    "lightblue": (173, 216, 230),
    "lightgreen":(144, 238, 144),
    "lightgrey": (211, 211, 211),
    "lightgray": (211, 211, 211),
    "navy":      (  0,   0, 128),
    "teal":      (  0, 128, 128),
    "lime":      (  0, 255,   0),
    "maroon":    (128,   0,   0),
    "olive":     (128, 128,   0),
    "aqua":      (  0, 255, 255),
    "fuchsia":   (255,   0, 255),
    "transparent":(  0,   0,   0),
}

# Default fallback values for optional element fields
ELEMENT_DEFAULTS: JsonDict = {
    "label":       "",
    "color":       [100, 100, 100],
    "hover_color": None,          # computed later when None
    "font_size":   24,
    "value":       50,
    "max_value":   100,
    "alpha":       255,
}

# Indentation helpers
I1 = "    "   # 4 spaces  (class body)
I2 = "        "  # 8 spaces  (method body)
I3 = "            "  # 12 spaces (nested block)
I4 = "                "  # 16 spaces


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _resolve_color(raw: Any) -> Tuple[int, int, int]:
    """Resolve a JSON colour value to an (R, G, B) tuple.

    Args:
        raw: Either a list/tuple of three ints ``[R, G, B]``,
             or a colour name string (e.g. ``"red"``).

    Returns:
        An ``(R, G, B)`` tuple with each component in ``[0, 255]``.

    Raises:
        ValueError: If *raw* cannot be interpreted as a colour.
    """
    if isinstance(raw, (list, tuple)):
        if len(raw) != 3:
            raise ValueError(f"Colour list must have exactly 3 components, got {raw!r}")
        r, g, b = int(raw[0]), int(raw[1]), int(raw[2])
        for name, val in (("R", r), ("G", g), ("B", b)):
            if not 0 <= val <= 255:
                raise ValueError(f"Colour component {name}={val} out of range [0, 255]")
        return (r, g, b)
    if isinstance(raw, str):
        key = raw.strip().lower()
        if key in NAMED_COLORS:
            return NAMED_COLORS[key]
        raise ValueError(f"Unknown colour name {raw!r}. Supported names: {sorted(NAMED_COLORS)}")
    raise ValueError(f"Colour must be a [R,G,B] list or a colour name string, got {type(raw).__name__!r}")


def _safe_identifier(text: str) -> str:
    """Convert an arbitrary string to a valid Python identifier.

    Args:
        text: Input string.

    Returns:
        A string that is a valid Python identifier.
    """
    ident = re.sub(r"[^0-9A-Za-z_]", "_", text)
    if ident and ident[0].isdigit():
        ident = "_" + ident
    return ident or "_element"


def _brighten(color: Tuple[int, int, int], amount: int = 40) -> Tuple[int, int, int]:
    """Return a brightened version of *color* by adding *amount* to each channel.

    Args:
        color:  Original (R, G, B) tuple.
        amount: Amount to add to each component (clamped to 255).

    Returns:
        Brightened (R, G, B) tuple.
    """
    return tuple(min(255, c + amount) for c in color)  # type: ignore[return-value]


def _class_name_from_string(raw: str) -> str:
    """Turn an arbitrary string into a PascalCase Python class name.

    Consecutive uppercase sequences (acronyms such as "HUD") are preserved
    as-is; only fully-lowercase words are capitalised.

    Args:
        raw: Raw string such as ``"game_hud"`` or ``"GameHUD"``.

    Returns:
        PascalCase class name, e.g. ``"GameHUD"``.
    """
    # Split on non-alphanumeric separators (underscores, spaces, hyphens …)
    words = re.split(r"[^A-Za-z0-9]+", raw)
    result_parts: List[str] = []
    for w in words:
        if not w:
            continue
        # Fully uppercase token (acronym like "HUD") → keep as-is
        # Mixed-case token that already starts uppercase (e.g. "GameHUD") → keep as-is
        # Lowercase token → capitalise first letter
        if w[0].isupper():
            result_parts.append(w)   # already starts capitalised – preserve
        else:
            result_parts.append(w.capitalize())
    return "".join(result_parts) or "GeneratedUI"


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

class SchemaValidationError(ValueError):
    """Raised when the input JSON does not conform to the expected schema."""


def validate_schema(data: JsonDict) -> None:
    """Validate the top-level structure of the UI layout JSON.

    Args:
        data: The parsed JSON object.

    Raises:
        SchemaValidationError: If a required field is missing or has the
            wrong type, or if any element is malformed.
    """
    # Top-level required keys
    for key in ("elements",):
        if key not in data:
            raise SchemaValidationError(f"Missing required top-level key: '{key}'")

    if not isinstance(data["elements"], list):
        raise SchemaValidationError("'elements' must be a JSON array.")

    seen_ids: set = set()
    for idx, el in enumerate(data["elements"]):
        prefix = f"Element[{idx}]"

        if not isinstance(el, dict):
            raise SchemaValidationError(f"{prefix}: each element must be a JSON object.")

        for field in REQUIRED_ELEMENT_FIELDS:
            if field not in el:
                raise SchemaValidationError(
                    f"{prefix}: missing required field '{field}'. "
                    f"Required fields: {REQUIRED_ELEMENT_FIELDS}"
                )

        el_id = el["id"]
        if not isinstance(el_id, str) or not el_id.strip():
            raise SchemaValidationError(f"{prefix}: 'id' must be a non-empty string.")
        if el_id in seen_ids:
            raise SchemaValidationError(f"{prefix}: duplicate element id '{el_id}'.")
        seen_ids.add(el_id)

        for dim in ("x", "y", "w", "h"):
            if not isinstance(el[dim], (int, float)):
                raise SchemaValidationError(
                    f"{prefix} id='{el_id}': field '{dim}' must be numeric, "
                    f"got {type(el[dim]).__name__!r}."
                )
            if el[dim] < 0:
                raise SchemaValidationError(
                    f"{prefix} id='{el_id}': field '{dim}' must be >= 0."
                )

        if "color" in el:
            try:
                _resolve_color(el["color"])
            except ValueError as exc:
                raise SchemaValidationError(
                    f"{prefix} id='{el_id}': invalid 'color': {exc}"
                ) from exc

        el_type = el.get("type", "")
        if el_type not in SUPPORTED_TYPES:
            logger.warning(
                "Element id='%s' has unknown type '%s'; a placeholder will be emitted.",
                el_id, el_type,
            )

    logger.info("Schema validation passed (%d elements).", len(data["elements"]))


# ---------------------------------------------------------------------------
# Per-element code snippets
# ---------------------------------------------------------------------------

class ElementSnippets:
    """Namespace for per-element-type code-generation helpers.

    Every public method returns a list of source-code lines (without trailing
    newlines) that belong inside a particular method body.

    Args:
        el:    The element dict (after defaults have been applied).
        indent: Leading indentation for each returned line.
    """

    def __init__(self, el: JsonDict, indent: str = I2) -> None:
        self._el = el
        self._ind = indent

    # ------------------------------------------------------------------
    # __init__ snippets
    # ------------------------------------------------------------------

    def init_lines(self) -> List[str]:
        """Generate ``__init__`` lines for this element."""
        el = self._el
        eid = el["id"]
        x, y, w, h = int(el["x"]), int(el["y"]), int(el["w"]), int(el["h"])
        color = _resolve_color(el.get("color", ELEMENT_DEFAULTS["color"]))
        font_size = int(el.get("font_size", ELEMENT_DEFAULTS["font_size"]))
        el_type = el["type"]

        lines: List[str] = []
        p = self._ind

        lines.append(f"{p}# --- {el_type}: {eid} ---")
        lines.append(f"{p}self.rect_{eid} = pygame.Rect({x}, {y}, {w}, {h})")
        lines.append(f"{p}self.color_{eid} = {color!r}")

        if el_type == "button":
            hover_raw = el.get("hover_color") or _brighten(color)
            hover_color = _resolve_color(hover_raw)
            lines.append(f"{p}self.hover_color_{eid} = {hover_color!r}")
            lines.append(f"{p}self.hovered_{eid}: bool = False")
            lines.append(f"{p}self.clicked_{eid}: bool = False")
            lines.append(f"{p}self.font_{eid} = pygame.font.SysFont('segoeui', {font_size})")

        elif el_type in ("health_bar", "progress_bar"):
            value = float(el.get("value", ELEMENT_DEFAULTS["value"]))
            max_value = float(el.get("max_value", ELEMENT_DEFAULTS["max_value"]))
            lines.append(f"{p}self.value_{eid}: float = {value}")
            lines.append(f"{p}self.max_value_{eid}: float = {max_value}")
            lines.append(f"{p}self.font_{eid} = pygame.font.SysFont('segoeui', {font_size})")

        elif el_type == "text_label":
            lines.append(f"{p}self.font_{eid} = pygame.font.SysFont('segoeui', {font_size})")
            label = el.get("label", "")
            lines.append(f"{p}self.text_{eid}: str = {label!r}")

        elif el_type == "icon":
            lines.append(f"{p}self.font_{eid} = pygame.font.SysFont('segoeui', {font_size})")

        # panel: just rect + color (already set above)
        lines.append("")
        return lines

    # ------------------------------------------------------------------
    # draw snippets
    # ------------------------------------------------------------------

    def draw_lines(self) -> List[str]:
        """Generate ``draw()`` lines for this element."""
        el = self._el
        eid = el["id"]
        el_type = el["type"]
        label = el.get("label", "")
        p = self._ind

        lines: List[str] = []
        lines.append(f"{p}# Draw {el_type}: {eid}")

        if el_type == "button":
            lines += self._draw_button(eid, label, p)

        elif el_type in ("health_bar", "progress_bar"):
            lines += self._draw_bar(eid, label, el_type, p)

        elif el_type == "text_label":
            lines += self._draw_text_label(eid, p)

        elif el_type == "panel":
            lines += self._draw_panel(eid, p)

        elif el_type == "icon":
            lines += self._draw_icon(eid, label, p)

        else:
            # Unknown type – emit a visible placeholder
            lines.append(f"{p}# UNKNOWN type '{el_type}' – placeholder")
            lines.append(f"{p}pygame.draw.rect(surface, (200, 50, 200), self.rect_{eid})")
            if label:
                lines.append(f"{p}_lbl = pygame.font.SysFont('segoeui', 20).render({label!r}, True, (255,255,255))")
                lines.append(f"{p}surface.blit(_lbl, self.rect_{eid}.topleft)")

        lines.append("")
        return lines

    @staticmethod
    def _draw_button(eid: str, label: str, p: str) -> List[str]:
        lines = [
            f"{p}# Hover tint",
            f"{p}_btn_color = self.hover_color_{eid} if self.hovered_{eid} else self.color_{eid}",
            f"{p}pygame.draw.rect(surface, _btn_color, self.rect_{eid}, border_radius=6)",
            f"{p}pygame.draw.rect(surface, (255, 255, 255), self.rect_{eid}, 2, border_radius=6)",
        ]
        if label:
            lines += [
                f"{p}_txt = self.font_{eid}.render({label!r}, True, (255, 255, 255))",
                f"{p}_txt_r = _txt.get_rect(center=self.rect_{eid}.center)",
                f"{p}surface.blit(_txt, _txt_r)",
            ]
        return lines

    @staticmethod
    def _draw_bar(eid: str, label: str, bar_type: str, p: str) -> List[str]:
        lines = [
            f"{p}# Background track",
            f"{p}pygame.draw.rect(surface, (50, 50, 50), self.rect_{eid}, border_radius=4)",
            f"{p}# Filled portion",
            f"{p}_ratio_{eid} = max(0.0, min(1.0, self.value_{eid} / max(1, self.max_value_{eid})))",
            f"{p}_fill_w_{eid} = int(self.rect_{eid}.width * _ratio_{eid})",
            f"{p}if _fill_w_{eid} > 0:",
            f"{p}    _fill_rect_{eid} = pygame.Rect(",
            f"{p}        self.rect_{eid}.x, self.rect_{eid}.y,",
            f"{p}        _fill_w_{eid}, self.rect_{eid}.height",
            f"{p}    )",
            f"{p}    pygame.draw.rect(surface, self.color_{eid}, _fill_rect_{eid}, border_radius=4)",
            f"{p}# Border",
            f"{p}pygame.draw.rect(surface, (200, 200, 200), self.rect_{eid}, 2, border_radius=4)",
        ]
        if label:
            lines += [
                f"{p}# Label + value",
                f"{p}_pct_{eid} = int(_ratio_{eid} * 100)",
                f"{p}_bar_txt = self.font_{eid}.render(",
                f"{p}    f\"{label}: {{self.value_{eid}:.0f}} / {{self.max_value_{eid}:.0f}}\",",
                f"{p}    True, (255, 255, 255)",
                f"{p})",
                f"{p}_bar_txt_r = _bar_txt.get_rect(midleft=(self.rect_{eid}.x + 4, self.rect_{eid}.centery))",
                f"{p}surface.blit(_bar_txt, _bar_txt_r)",
            ]
        return lines

    @staticmethod
    def _draw_text_label(eid: str, p: str) -> List[str]:
        return [
            f"{p}_surf_{eid} = self.font_{eid}.render(self.text_{eid}, True, self.color_{eid})",
            f"{p}_rect_{eid} = _surf_{eid}.get_rect(center=self.rect_{eid}.center)",
            f"{p}surface.blit(_surf_{eid}, _rect_{eid})",
        ]

    @staticmethod
    def _draw_panel(eid: str, p: str) -> List[str]:
        return [
            f"{p}pygame.draw.rect(surface, self.color_{eid}, self.rect_{eid}, border_radius=8)",
            f"{p}pygame.draw.rect(surface, (180, 180, 180), self.rect_{eid}, 2, border_radius=8)",
        ]

    @staticmethod
    def _draw_icon(eid: str, label: str, p: str) -> List[str]:
        lines = [
            f"{p}pygame.draw.rect(surface, self.color_{eid}, self.rect_{eid}, border_radius=4)",
            f"{p}pygame.draw.rect(surface, (220, 220, 220), self.rect_{eid}, 2, border_radius=4)",
        ]
        if label:
            lines += [
                f"{p}_icon_txt = self.font_{eid}.render({label!r}, True, (255, 255, 255))",
                f"{p}_icon_txt_r = _icon_txt.get_rect(center=self.rect_{eid}.center)",
                f"{p}surface.blit(_icon_txt, _icon_txt_r)",
            ]
        return lines

    # ------------------------------------------------------------------
    # update snippets
    # ------------------------------------------------------------------

    def update_lines(self) -> List[str]:
        """Generate ``update()`` lines for this element."""
        el = self._el
        eid = el["id"]
        el_type = el["type"]
        p = self._ind

        lines: List[str] = []

        if el_type == "button":
            lines.append(f"{p}# Update {el_type}: {eid}")
            lines.append(f"{p}self.hovered_{eid} = self.rect_{eid}.collidepoint(mouse_pos)")
            lines.append(f"{p}if self.hovered_{eid}:")
            lines.append(f"{p}    for _evt in events:")
            lines.append(f"{p}        if _evt.type == pygame.MOUSEBUTTONDOWN and _evt.button == 1:")
            lines.append(f"{p}            self.clicked_{eid} = not self.clicked_{eid}")
            lines.append("")

        return lines

    # ------------------------------------------------------------------
    # get_state snippets
    # ------------------------------------------------------------------

    def state_key_value(self) -> Optional[Tuple[str, str]]:
        """Return ``(key, value_expr)`` pair for the state dict, or ``None``.

        Returns:
            A tuple of (state dict key string, Python expression string)
            to include in the ``get_state()`` return dict, or ``None`` if
            this element type exposes no state.
        """
        el = self._el
        eid = el["id"]
        el_type = el["type"]

        if el_type == "button":
            return (eid, f"self.clicked_{eid}")
        if el_type in ("health_bar", "progress_bar"):
            return (eid, f"{{'value': self.value_{eid}, 'max_value': self.max_value_{eid}}}")
        if el_type == "text_label":
            return (eid, f"self.text_{eid}")
        return None


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------

class CodeGeneratorEngine:
    """Transforms a UI-layout JSON into a ready-to-run Pygame Python module.

    Example:
        >>> engine = CodeGeneratorEngine()
        >>> engine.run("layout.json", "generated_ui.py")

    Attributes:
        source_path: Path to the JSON that was last loaded (set by
            :py:meth:`load_json`).
    """

    def __init__(self) -> None:
        self.source_path: Optional[str] = None

    # ------------------------------------------------------------------
    # I/O helpers
    # ------------------------------------------------------------------

    def load_json(self, path: str) -> JsonDict:
        """Load and parse the layout JSON from *path*.

        Args:
            path: File-system path to a ``.json`` file.

        Returns:
            Parsed JSON object as a Python dict.

        Raises:
            FileNotFoundError: If *path* does not exist.
            json.JSONDecodeError: If the file contains invalid JSON.
        """
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Input file not found: '{path}'")
        with open(path, "r", encoding="utf-8") as fh:
            data: JsonDict = json.load(fh)
        self.source_path = os.path.abspath(path)
        logger.info("Loaded JSON from '%s'.", path)
        return data

    def save(self, code: str, output_path: str) -> None:
        """Write *code* to *output_path*, creating parent directories as needed.

        Args:
            code:        The Python source code string to write.
            output_path: Destination file path.
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as fh:
            fh.write(code)
        logger.info("Generated code written to '%s'.", output_path)

    # ------------------------------------------------------------------
    # Defaults application
    # ------------------------------------------------------------------

    def _apply_defaults(self, el: JsonDict) -> JsonDict:
        """Return a copy of *el* with all optional fields filled to defaults.

        Args:
            el: Raw element dict from the JSON.

        Returns:
            New dict with defaults applied (original is not mutated).
        """
        result = dict(ELEMENT_DEFAULTS)
        result.update(el)
        # Ensure id is a valid Python identifier
        result["id"] = _safe_identifier(str(result["id"]))
        return result

    # ------------------------------------------------------------------
    # Code generation – sub-sections
    # ------------------------------------------------------------------

    def generate_imports(self) -> str:
        """Return the import block for the generated file.

        Returns:
            Multi-line import string.
        """
        return textwrap.dedent("""\
            import pygame
            from typing import Any, Dict, List, Optional
        """)

    def generate_init_method(self, elements: List[JsonDict]) -> str:
        """Generate the ``__init__`` method body.

        Args:
            elements: List of element dicts (defaults already applied).

        Returns:
            Source code string for the ``__init__`` method.
        """
        lines: List[str] = []
        lines.append(f"{I1}def __init__(self) -> None:")
        lines.append(f'{I2}"""Initialise all Pygame Rects, fonts, colours and state variables."""')
        lines.append(f"{I2}pygame.font.init()")
        lines.append("")

        for el in elements:
            snip = ElementSnippets(el, indent=I2)
            lines.extend(snip.init_lines())

        return "\n".join(lines)

    def generate_draw_method(self, elements: List[JsonDict]) -> str:
        """Generate the ``draw(surface)`` method body.

        Args:
            elements: List of element dicts (defaults already applied).

        Returns:
            Source code string for the ``draw`` method.
        """
        lines: List[str] = []
        lines.append(f"{I1}def draw(self, surface: pygame.Surface) -> None:")
        lines.append(f'{I2}"""Draw all UI elements onto *surface*.')
        lines.append("")
        lines.append(f"{I2}Args:")
        lines.append(f"{I2}    surface: The Pygame surface to draw onto (typically the display).")
        lines.append(f'{I2}"""')
        lines.append("")

        for el in elements:
            snip = ElementSnippets(el, indent=I2)
            lines.extend(snip.draw_lines())

        return "\n".join(lines)

    def generate_update_method(self, elements: List[JsonDict]) -> str:
        """Generate the ``update(events)`` method body.

        Args:
            elements: List of element dicts (defaults already applied).

        Returns:
            Source code string for the ``update`` method.
        """
        lines: List[str] = []
        lines.append(f"{I1}def update(self, events: List[pygame.event.Event]) -> None:")
        lines.append(f'{I2}"""Process input events and update element states.')
        lines.append("")
        lines.append(f"{I2}Args:")
        lines.append(f"{I2}    events: The list returned by ``pygame.event.get()``.")
        lines.append(f'{I2}"""')
        lines.append(f"{I2}mouse_pos = pygame.mouse.get_pos()")
        lines.append("")

        interactive = [el for el in elements if el["type"] == "button"]
        if not interactive:
            lines.append(f"{I2}# No interactive elements in this layout.")
            lines.append(f"{I2}pass")
        else:
            for el in interactive:
                snip = ElementSnippets(el, indent=I2)
                lines.extend(snip.update_lines())

        return "\n".join(lines)

    def generate_state_method(self, elements: List[JsonDict]) -> str:
        """Generate the ``get_state()`` method body.

        Args:
            elements: List of element dicts (defaults already applied).

        Returns:
            Source code string for the ``get_state`` method.
        """
        lines: List[str] = []
        lines.append(f"{I1}def get_state(self) -> Dict[str, Any]:")
        lines.append(f'{I2}"""Return a snapshot of the current UI state.')
        lines.append("")
        lines.append(f"{I2}Returns:")
        lines.append(f"{I2}    A dictionary mapping element ids to their state.  Buttons map to")
        lines.append(f"{I2}    ``bool`` (toggled), bars map to ``{{'value': …, 'max_value': …}}``,")
        lines.append(f"{I2}    text labels map to the current text string.")
        lines.append(f'{I2}"""')
        lines.append(f"{I2}return {{")

        for el in elements:
            snip = ElementSnippets(el, indent=I2)
            kv = snip.state_key_value()
            if kv is not None:
                key, val = kv
                lines.append(f"{I3}{key!r}: {val},")

        lines.append(f"{I2}}}")

        return "\n".join(lines)

    def generate_demo_block(
        self,
        class_name: str,
        screen_w: int,
        screen_h: int,
        bg_color: Tuple[int, int, int],
    ) -> str:
        """Generate the ``if __name__ == '__main__':`` standalone demo block.

        Args:
            class_name: Name of the generated class.
            screen_w:   Width of the demo window.
            screen_h:   Height of the demo window.
            bg_color:   Background fill colour as (R, G, B).

        Returns:
            Source code string for the demo entry-point.
        """
        return textwrap.dedent(f"""\
            if __name__ == "__main__":
                \"\"\"Standalone demo – run this file directly to preview the UI.\"\"\"
                pygame.init()
                screen = pygame.display.set_mode(({screen_w}, {screen_h}))
                pygame.display.set_caption("{class_name} – Preview")
                clock = pygame.time.Clock()
                ui = {class_name}()

                running = True
                while running:
                    events = pygame.event.get()
                    for event in events:
                        if event.type == pygame.QUIT:
                            running = False
                        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                            running = False

                    screen.fill({bg_color!r})
                    ui.update(events)
                    ui.draw(screen)
                    pygame.display.flip()
                    clock.tick(60)

                # Print final state before exit
                print("Final UI state:", ui.get_state())
                pygame.quit()
        """)

    def generate_class_code(self, data: JsonDict, class_name: str) -> str:
        """Assemble the complete Python source for the UI class.

        Args:
            data:       Validated and parsed JSON layout dict.
            class_name: Python class name to use for the generated class.

        Returns:
            Full Python source code as a string.
        """
        # Resolve optional top-level fields
        screen_w = int(data.get("screen_width", 800))
        screen_h = int(data.get("screen_height", 600))
        raw_bg = data.get("background_color", [30, 30, 30])
        bg_color = _resolve_color(raw_bg)
        ui_name = data.get("ui_name", class_name)

        # Apply defaults to every element
        elements = [self._apply_defaults(el) for el in data["elements"]]

        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        source_ref = os.path.basename(self.source_path) if self.source_path else "unknown"

        # ---- file header ----
        header = textwrap.dedent(f"""\
            # =============================================================================
            # Auto-generated by CodeGeneratorEngine
            # Source  : {source_ref}
            # UI Name : {ui_name}
            # Class   : {class_name}
            # Generated: {timestamp}
            #
            # This file was produced automatically – manual edits may be overwritten.
            # To regenerate:  python code_generator.py --input <layout.json> --output <this file>
            # =============================================================================
        """)

        # ---- imports ----
        imports = self.generate_imports()

        # ---- class docstring ----
        class_doc = textwrap.dedent(f"""\
            class {class_name}:
            {I1}\"\"\"Auto-generated Pygame UI class for "{ui_name}".

            {I1}This class manages {len(elements)} UI element(s) defined in ``{source_ref}``.
            {I1}Screen dimensions: {screen_w} x {screen_h}.  Background: {bg_color!r}.

            {I1}Usage::

            {I1}    pygame.init()
            {I1}    screen = pygame.display.set_mode(({screen_w}, {screen_h}))
            {I1}    ui = {class_name}()

            {I1}    # Inside your game loop:
            {I1}    events = pygame.event.get()
            {I1}    ui.update(events)
            {I1}    ui.draw(screen)
            {I1}    state = ui.get_state()
            {I1}\"\"\"
        """)

        # ---- methods ----
        init_code = self.generate_init_method(elements)
        draw_code = self.generate_draw_method(elements)
        update_code = self.generate_update_method(elements)
        state_code = self.generate_state_method(elements)

        # ---- bar value setters (convenience) ----
        setter_lines = self._generate_bar_setters(elements)

        # ---- demo ----
        demo_code = self.generate_demo_block(class_name, screen_w, screen_h, bg_color)

        # ---- assemble ----
        sections = [
            header,
            imports,
            "",
            class_doc,
            "",
            init_code,
            "",
            draw_code,
            "",
            update_code,
            "",
            state_code,
        ]
        if setter_lines:
            sections += ["", setter_lines]

        sections += ["", "", demo_code]

        return "\n".join(sections) + "\n"

    def _generate_bar_setters(self, elements: List[JsonDict]) -> str:
        """Generate convenience setter methods for health/progress bars.

        Args:
            elements: List of element dicts with defaults applied.

        Returns:
            Source code for all setter methods, or empty string if none.
        """
        lines: List[str] = []
        for el in elements:
            if el["type"] not in ("health_bar", "progress_bar"):
                continue
            eid = el["id"]
            lines.append(f"{I1}def set_value_{eid}(self, value: float) -> None:")
            lines.append(f'{I2}"""Set current value for the "{eid}" bar.')
            lines.append("")
            lines.append(f"{I2}Args:")
            lines.append(f"{I2}    value: New value (clamped to [0, max_value]).")
            lines.append(f'{I2}"""')
            lines.append(f"{I2}self.value_{eid} = max(0.0, min(float(self.max_value_{eid}), float(value)))")
            lines.append("")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(
        self,
        input_path: str,
        output_path: str,
        class_name: Optional[str] = None,
    ) -> str:
        """Full pipeline: load → validate → generate → save.

        Args:
            input_path:  Path to the JSON layout file.
            output_path: Destination ``.py`` file path.
            class_name:  Override the generated class name.  If ``None``,
                         the name is taken from the ``"ui_name"`` field in
                         the JSON (PascalCase-normalised).

        Returns:
            The generated Python source code as a string.

        Raises:
            FileNotFoundError:    If *input_path* does not exist.
            SchemaValidationError: If the JSON fails schema validation.
        """
        data = self.load_json(input_path)
        validate_schema(data)

        if class_name is None:
            raw_name = data.get("ui_name", "GeneratedUI")
            class_name = _class_name_from_string(raw_name)

        logger.info("Generating class '%s' …", class_name)
        code = self.generate_class_code(data, class_name)
        self.save(code, output_path)
        logger.info("Done.  Output → '%s'.", output_path)
        return code


# ---------------------------------------------------------------------------
# Library-level convenience function
# ---------------------------------------------------------------------------

def generate_pygame_ui(
    json_path: str,
    output_path: str,
    class_name: Optional[str] = None,
) -> str:
    """Generate a Pygame UI module from a JSON layout file.

    This is the library-callable entry point.  Internally it creates a
    :py:class:`CodeGeneratorEngine` and calls :py:meth:`~CodeGeneratorEngine.run`.

    Args:
        json_path:   Path to the input JSON layout file.
        output_path: Destination ``.py`` file path.
        class_name:  Optional override for the generated class name.

    Returns:
        The generated Python source code as a string.

    Example:
        >>> generate_pygame_ui("layout.json", "ui.py", class_name="GameHUD")
    """
    engine = CodeGeneratorEngine()
    return engine.run(json_path, output_path, class_name=class_name)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="code_generator",
        description=(
            "Pygame UI Code Generator Engine – "
            "converts a JSON layout file into a Pygame Python module."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python code_generator.py --input layout.json --output generated_ui.py
              python code_generator.py --input hud.json --output hud_ui.py --class-name GameHUD
        """),
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        metavar="JSON_FILE",
        help="Path to the input UI layout JSON file.",
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        metavar="PY_FILE",
        help="Destination path for the generated Python module.",
    )
    parser.add_argument(
        "--class-name", "-c",
        default=None,
        metavar="CLASS_NAME",
        dest="class_name",
        help=(
            "Python class name for the generated UI class.  "
            "Defaults to the 'ui_name' field in the JSON."
        ),
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Exit code (0 = success, non-zero = error).
    """
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        engine = CodeGeneratorEngine()
        engine.run(args.input, args.output, class_name=args.class_name)
        print(f"✓ Generated '{args.output}' successfully.")
        return 0
    except FileNotFoundError as exc:
        logger.error("File not found: %s", exc)
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except SchemaValidationError as exc:
        logger.error("Schema validation failed: %s", exc)
        print(f"SCHEMA ERROR: {exc}", file=sys.stderr)
        return 3
    except json.JSONDecodeError as exc:
        logger.error("Invalid JSON: %s", exc)
        print(f"JSON ERROR: {exc}", file=sys.stderr)
        return 4
    except Exception as exc:  # pylint: disable=broad-except
        logger.exception("Unexpected error: %s", exc)
        print(f"UNEXPECTED ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
