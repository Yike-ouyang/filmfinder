from __future__ import annotations

import io
import zipfile
from pathlib import PurePosixPath, PureWindowsPath

from .exceptions import APIError


def _is_unsafe_name(name):
    if not name or "\x00" in name:
        return True
    if "\\" in name:
        return True
    if name.startswith("/") or name.startswith("\\"):
        return True
    posix = PurePosixPath(name)
    windows = PureWindowsPath(name)
    if posix.is_absolute() or windows.is_absolute():
        return True
    if any(part == ".." for part in posix.parts):
        return True
    if any(part == ".." for part in windows.parts):
        return True
    return False


def validate_archive(data, max_compressed_bytes, max_entries, max_uncompressed_bytes):
    if not data:
        raise APIError(400, "empty_upload", "The uploaded archive is empty.")
    if len(data) > int(max_compressed_bytes):
        raise APIError(
            413,
            "upload_too_large",
            "The uploaded archive exceeds the size limit.",
        )
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except (zipfile.BadZipFile, OSError) as error:
        raise APIError(
            400,
            "invalid_zip",
            "The uploaded file is not a valid ZIP archive.",
        ) from error
    try:
        infos = archive.infolist()
        if len(infos) > int(max_entries):
            raise APIError(
                413,
                "too_many_entries",
                "The archive contains too many files.",
            )
        total_size = 0
        names = set()
        for info in infos:
            if info.flag_bits & 0x1:
                raise APIError(
                    400,
                    "encrypted_zip",
                    "Encrypted ZIP archives are not supported.",
                )
            if _is_unsafe_name(info.filename):
                raise APIError(
                    400,
                    "unsafe_zip_path",
                    "The archive contains an unsafe path.",
                )
            if info.filename in names:
                raise APIError(
                    400,
                    "duplicate_zip_entry",
                    "The archive contains duplicate entries.",
                )
            names.add(info.filename)
            total_size += int(info.file_size)
            if total_size > int(max_uncompressed_bytes):
                raise APIError(
                    413,
                    "uncompressed_too_large",
                    "The uncompressed archive exceeds the size limit.",
                )
        if "ratings.csv" not in names:
            raise APIError(
                400,
                "missing_ratings",
                "The archive does not contain ratings.csv.",
            )
        bad_member = archive.testzip()
        if bad_member is not None:
            raise APIError(
                400,
                "corrupted_zip",
                "The archive contains a corrupted file.",
            )
    finally:
        archive.close()
