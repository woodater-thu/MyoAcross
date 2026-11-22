"""
Pure NumPy/scipy reader for MATLAB MCOS ``table`` payloads in MAT v5 files.

The Gait120 ``EMG/ProcessedData.mat`` files store every table as an opaque MATLAB object (MCOS).
``scipy.io.loadmat`` decodes the surrounding struct tree but returns each table as an unresolvable ``MatlabOpaque`` reference.
This module resolves those references against the file's embedded ``FileWrapper`` subsystem so the runtime never needs MATLAB.

Verified format facts this implementation relies on (checked against real Gait120 subjects):

- ``__function_workspace__`` holds two matrices: a ``1x1`` struct whose single field ``MCOS`` is the ``FileWrapper__`` opaque,
  followed by a tiny trailer.
- The FileWrapper opaque carries one cell per subsystem entry:
  cell 0 is the serialized metadata blob, cells 1.. are the referenced values.
- Every table reference payload is exactly ``[0xDCE00000, 2, 1, 1, object_id, 1]``.
- Each table object occupies seven consecutive FileWrapper cells starting at ``2 + 7 * (object_id - 1)``,
  ordered ``(data, ndims, nrows, rownames, nvars, variable_names, properties)``.

Any deviation raises immediately instead of producing wrong numbers.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat
from scipy.io.matlab._mio5 import MatFile5Reader
from scipy.io.matlab import MatReadError

_MCOR_MAGIC = 0xDD000000
_REFERENCE_TAIL = (2, 1, 1)
_OBJECT_BASE_CELL = 2
_TABLE_CELL_COUNT = 7


class McosDecodeError(ValueError):
    """Raised when a MAT v5 file violates the expected MCOS layout."""


def resolve_tables(path: Path) -> dict[str, Any]:
    """
    Load a MAT v5 file and replace every MCOS table leaf with a plain dict.

    Returns the usual ``loadmat`` mapping with every ``MatlabOpaque`` replaced by ``{"values": ndarray (nrows, nvars) float64, "variable_names": tuple[str, ...]}``.
    """

    source = Path(path)
    try:
        document = loadmat(source)
    except (OSError, ValueError, IndexError, MatReadError) as exc:
        raise McosDecodeError(f"{source}: unreadable MAT file ({exc})") from exc
    if _has_mcos_subsystem(document):
        workspace = document["__function_workspace__"]
        cells = _file_wrapper_cells(raw=source.read_bytes(), workspace=np.asarray(workspace).tobytes())
    else:
        # Plain MAT v5 files carry no MCOS objects; keep the unified dict shape.
        cells = ()
    return {
        key: _convert(value, cells, source, key)
        for key, value in document.items()
        if not key.startswith("__")
    }


def _has_mcos_subsystem(document: dict[str, Any]) -> bool:
    """Return whether the file embeds an MCOS function-workspace subsystem."""

    workspace = document.get("__function_workspace__")
    return (
        isinstance(workspace, np.ndarray)
        and workspace.dtype == np.dtype(np.uint8)
        and workspace.size > 0
    )


def _file_wrapper_cells(raw: bytes, workspace: bytes) -> tuple[Any, ...]:
    """Return every FileWrapper cell of the function-workspace subsystem."""

    reader = _nested_reader(raw, workspace[8:])
    try:
        header, _ = reader.read_var_header()
        top = reader.read_var_array(header)
    except Exception as exc:  # pragma: no cover - defensive
        raise McosDecodeError(f"unreadable function workspace: {exc}") from exc
    names = getattr(top.dtype, "names", None)
    if names != ("MCOS",):
        raise McosDecodeError("function workspace must be a struct with a single MCOS field")
    reference = _unwrap(top.ravel()[0]["MCOS"])
    if not isinstance(reference, np.ndarray) or "arr" not in getattr(reference.dtype, "names", ()):
        raise McosDecodeError("MCOS field must decode to a FileWrapper opaque reference")
    cells = _unwrap(reference["arr"])
    if not isinstance(cells, np.ndarray) or cells.dtype != object:
        raise McosDecodeError("FileWrapper payload must be an object array of cells")
    return tuple(cells.ravel())


def _nested_reader(raw: bytes, stream: bytes) -> MatFile5Reader:
    """Open a MAT5 reader over an embedded element stream lacking a file header."""

    if len(raw) < 128:
        raise McosDecodeError("MAT file too short")
    forged_header = raw[:116] + b"\x00" * 8 + raw[124:128]
    reader = MatFile5Reader(BytesIO(forged_header + stream))
    reader.read_file_header()
    reader.initialize_read()
    return reader


def _unwrap(value: Any) -> Any:
    """Strip singleton object-array wrappers around a value."""

    current = value
    while isinstance(current, np.ndarray) and current.dtype == object and current.size == 1:
        current = current.ravel()[0]
    return current


def _convert(
    value: Any,
    cells: tuple[Any, ...],
    source: Path,
    path: str,
) -> Any:
    """Recursively convert a loadmat subtree, resolving MCOS references."""

    if isinstance(value, np.ndarray) and value.dtype.names and set(value.dtype.names) == {"s0", "s1", "s2", "arr"}:
        return _resolve_reference(np.asarray(value).ravel()[0], cells, source, path)
    if isinstance(value, np.ndarray) and value.dtype == object:
        return [_convert(item, cells, source, f"{path}[{index}]") for index, item in enumerate(value.ravel())]
    if isinstance(value, np.ndarray) and value.dtype.names:
        record = value.ravel()[0]
        return {name: _convert(record[name], cells, source, f"{path}.{name}") for name in value.dtype.names}
    return value


def _resolve_reference(leaf: Any, cells: tuple[Any, ...], source: Path, path: str) -> dict[str, Any]:
    """Resolve one MCOS table reference to its numeric content."""

    if not isinstance(leaf, (np.ndarray, np.void)) or "arr" not in getattr(leaf.dtype, "names", ()):
        raise McosDecodeError(f"{source}:{path}: opaque reference has unexpected layout")
    payload = np.asarray(_unwrap(leaf["arr"])).ravel()
    if payload.size != 6:
        raise McosDecodeError(f"{source}:{path}: MCOS reference payload must hold six words")
    words = tuple(int(word) for word in payload)
    magic, segment_count, first_id, first_pos, object_id, second_pos = words
    if magic != _MCOR_MAGIC or (segment_count, first_id, first_pos) != _REFERENCE_TAIL or second_pos != 1:
        raise McosDecodeError(f"{source}:{path}: unsupported MCOS reference payload {words}")
    start = _OBJECT_BASE_CELL + _TABLE_CELL_COUNT * (object_id - 1)
    if start + _TABLE_CELL_COUNT > len(cells):
        raise McosDecodeError(f"{source}:{path}: object id {object_id} outside FileWrapper")
    block = [_unwrap(cells[start + offset]) for offset in range(_TABLE_CELL_COUNT)]
    data, dims, rows, _rownames, variables, variable_names, _properties = block
    values = _table_values(data, source, path)
    count_dims = _scalar_int(dims, "ndims", source, path)
    count_rows = _scalar_int(rows, "nrows", source, path)
    count_variables = _scalar_int(variables, "nvars", source, path)
    if count_dims != 2:
        raise McosDecodeError(f"{source}:{path}: table ndims must be 2")
    names = _table_variable_names(variable_names, source, path)
    if count_variables != values.shape[1] or len(names) != values.shape[1]:
        raise McosDecodeError(f"{source}:{path}: table nvars disagrees with its data columns")
    if count_rows != values.shape[0]:
        raise McosDecodeError(f"{source}:{path}: table nrows disagrees with its data columns")
    return {"values": values, "variable_names": names}


def _table_values(data: Any, source: Path, path: str) -> np.ndarray:
    columns = np.asarray(data)
    if columns.dtype != object or columns.size == 0:
        raise McosDecodeError(f"{source}:{path}: table data must be a list of column vectors")
    stacked: list[np.ndarray] = []
    for index, item in enumerate(columns.ravel()):
        column = np.asarray(item, dtype=np.float64)
        if column.ndim != 2:
            column = column.reshape(-1, 1)
        if column.shape[1] != 1:
            raise McosDecodeError(f"{source}:{path}: table column {index} must be a single vector")
        stacked.append(column[:, 0])
    if not stacked:
        raise McosDecodeError(f"{source}:{path}: table has no data columns")
    lengths = {column.shape[0] for column in stacked}
    if len(lengths) != 1:
        raise McosDecodeError(f"{source}:{path}: table columns have inconsistent lengths {sorted(lengths)}")
    values = np.column_stack(stacked)
    if not np.isfinite(values).all():
        raise McosDecodeError(f"{source}:{path}: table values must be finite")
    return values


def _table_variable_names(variable_names: Any, source: Path, path: str) -> tuple[str, ...]:
    array = np.asarray(variable_names)
    if array.dtype != object or array.size == 0:
        raise McosDecodeError(f"{source}:{path}: table variable_names must be a non-empty list")
    names: list[str] = []
    for item in array.ravel():
        inner = _unwrap(item)
        if isinstance(inner, np.ndarray) and inner.size == 1:
            names.append(str(inner.ravel()[0]))
        elif isinstance(inner, str):
            names.append(inner)
        else:
            raise McosDecodeError(f"{source}:{path}: table variable_names entries must be strings")
    return tuple(names)


def _scalar_int(value: Any, label: str, source: Path, path: str) -> int:
    scalar = np.asarray(value)
    if scalar.size != 1:
        raise McosDecodeError(f"{source}:{path}: table {label} must be scalar")
    return int(scalar.ravel()[0])


__all__ = ["McosDecodeError", "resolve_tables"]
