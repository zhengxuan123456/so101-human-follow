"""Resolve Windows camera names to OpenCV DirectShow indices."""

from __future__ import annotations

import ctypes as ct
import os
from uuid import UUID


class _Guid(ct.Structure):
    _fields_ = [
        ("data1", ct.c_uint32),
        ("data2", ct.c_uint16),
        ("data3", ct.c_uint16),
        ("data4", ct.c_ubyte * 8),
    ]


class _Variant(ct.Structure):
    # The VARIANT value union includes pairs of pointers. Allocate its full size
    # on both Windows architectures, even though we only read the BSTR member.
    _fields_ = [("type", ct.c_uint16), ("reserved", ct.c_uint16 * 3), ("value", ct.c_void_p * 2)]


_SYSTEM_DEVICE_ENUM = _Guid.from_buffer_copy(UUID("62BE5D10-60EB-11D0-BD3B-00A0C911CE86").bytes_le)
_CREATE_DEVICE_ENUM = _Guid.from_buffer_copy(UUID("29840822-5B84-11D0-BD3B-00A0C911CE86").bytes_le)
_VIDEO_INPUT_CATEGORY = _Guid.from_buffer_copy(UUID("860BB310-5D01-11D0-BD3B-00A0C911CE86").bytes_le)
_PROPERTY_BAG = _Guid.from_buffer_copy(UUID("55272A00-42CB-11CE-8135-00AA004BB851").bytes_le)
_HRESULT = ct.c_int32
_OUT_INTERFACE = ct.POINTER(ct.c_void_p)
_CHANGED_APARTMENT_MODE = _HRESULT(0x80010106).value


def _method(interface: ct.c_void_p, slot: int, result: type, *args: type):
    table = ct.cast(interface, ct.POINTER(ct.POINTER(ct.c_void_p))).contents
    return ct.WINFUNCTYPE(result, ct.c_void_p, *args)(table[slot])


def _release(interface: ct.c_void_p) -> None:
    if interface:
        # IUnknown::Release
        _method(interface, 2, ct.c_uint32)(interface)


def _check(result: int, operation: str) -> None:
    if result < 0:
        raise RuntimeError(f"{operation}失败（Windows 0x{result & 0xffffffff:08X}）。")


def _friendly_name(moniker: ct.c_void_p, automation) -> str | None:
    bag = ct.c_void_p()
    # IMoniker::BindToStorage retrieves metadata without opening the camera.
    bind = _method(moniker, 9, _HRESULT, ct.c_void_p, ct.c_void_p, ct.POINTER(_Guid), _OUT_INTERFACE)
    result = bind(moniker, None, None, ct.byref(_PROPERTY_BAG), ct.byref(bag))
    if result < 0 or not bag:
        return None
    value = _Variant()
    automation.VariantInit(ct.byref(value))
    try:
        # IPropertyBag::Read. OpenCV also requires FriendlyName to open this index.
        read = _method(bag, 3, _HRESULT, ct.c_wchar_p, ct.POINTER(_Variant), ct.c_void_p)
        result = read(bag, "FriendlyName", ct.byref(value), None)
        if result == 0 and value.type == 8 and value.value[0]:  # VT_BSTR
            return ct.wstring_at(value.value[0]).strip() or None
        return None
    finally:
        automation.VariantClear(ct.byref(value))
        _release(bag)


def _camera_names_by_index() -> list[str | None]:
    if os.name != "nt":
        raise RuntimeError("按设备名称连接相机目前需要 Windows。")

    ole = ct.WinDLL("ole32")
    ole.CoInitializeEx.argtypes = (ct.c_void_p, ct.c_uint32)
    ole.CoInitializeEx.restype = _HRESULT
    ole.CoUninitialize.argtypes = ()
    ole.CoUninitialize.restype = None
    ole.CoCreateInstance.argtypes = (
        ct.POINTER(_Guid), ct.c_void_p, ct.c_uint32, ct.POINTER(_Guid), _OUT_INTERFACE,
    )
    ole.CoCreateInstance.restype = _HRESULT
    automation = ct.WinDLL("oleaut32")
    automation.VariantInit.argtypes = (ct.POINTER(_Variant),)
    automation.VariantInit.restype = None
    automation.VariantClear.argtypes = (ct.POINTER(_Variant),)
    automation.VariantClear.restype = _HRESULT

    initialized = ole.CoInitializeEx(None, 0)  # COINIT_MULTITHREADED
    if initialized != _CHANGED_APARTMENT_MODE:
        _check(initialized, "初始化 Windows 相机枚举")
    device_enum = ct.c_void_p()
    moniker_enum = ct.c_void_p()
    try:
        _check(ole.CoCreateInstance(
            ct.byref(_SYSTEM_DEVICE_ENUM), None, 1, ct.byref(_CREATE_DEVICE_ENUM), ct.byref(device_enum),
        ), "创建 Windows 相机枚举器")  # CLSCTX_INPROC_SERVER
        # ICreateDevEnum::CreateClassEnumerator
        enumerate_category = _method(device_enum, 3, _HRESULT, ct.POINTER(_Guid), _OUT_INTERFACE, ct.c_uint32)
        result = enumerate_category(device_enum, ct.byref(_VIDEO_INPUT_CATEGORY), ct.byref(moniker_enum), 0)
        _check(result, "读取 Windows 相机设备")
        if result != 0:
            return []  # S_FALSE when this device category is empty.

        # IEnumMoniker::Next. Keep unnamed entries so subsequent indices still
        # match videoInput::getDevice in OpenCV's CAP_DSHOW implementation.
        next_moniker = _method(moniker_enum, 3, _HRESULT, ct.c_uint32, _OUT_INTERFACE, ct.POINTER(ct.c_uint32))
        names = []
        while True:
            moniker = ct.c_void_p()
            result = next_moniker(moniker_enum, 1, ct.byref(moniker), None)
            _check(result, "读取 Windows 相机列表")
            if result != 0:
                return names
            try:
                names.append(_friendly_name(moniker, automation))
            finally:
                _release(moniker)
    finally:
        _release(moniker_enum)
        _release(device_enum)
        if initialized >= 0:
            ole.CoUninitialize()


def list_camera_names() -> list[str]:
    """List Windows camera names without opening a capture stream."""
    return [name for name in _camera_names_by_index() if name is not None]


def camera_index_by_name(name: str) -> int:
    """Find one camera by full name, ignoring case and surrounding whitespace."""
    name = name.strip()
    if not name:
        raise ValueError("请输入设备管理器中的相机名称。")
    names = _camera_names_by_index()
    matches = [index for index, current in enumerate(names) if current and current.casefold() == name.casefold()]
    if not matches:
        choices = "、".join(current for current in names if current)
        detail = f"当前可用相机：{choices}。" if choices else "当前没有检测到可用相机。"
        raise ValueError(f"找不到相机「{name}」。{detail}")
    if len(matches) != 1:
        raise ValueError(f"检测到多个名为「{name}」的相机。请仅保留其中一个后重试。")
    return matches[0]
