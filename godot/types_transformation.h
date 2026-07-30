#pragma once

#include <vector>
#include <godot_cpp/variant/packed_float32_array.hpp>

inline std::vector<float> packedArrayToVector(const godot::PackedFloat32Array& arr)
{
    return std::vector<float>(arr.ptr(), arr.ptr() + arr.size());
}

inline godot::PackedFloat32Array vectorToPackedArray(const std::vector<float>& vec)
{
    godot::PackedFloat32Array arr;
    arr.resize(vec.size());
    float* arr_ptr = arr.ptrw();
    if (!vec.empty())
        std::memcpy(arr.ptrw(), vec.data(), vec.size() * sizeof(float));
    return arr;
}

inline godot::TypedArray<godot::PackedFloat32Array> vectorVectorToArrayPackedArray(const std::vector<std::vector<float>>& vecVec)
{
    godot::TypedArray<godot::PackedFloat32Array> result;
    result.resize(vecVec.size());
    for (size_t i = 0; i < vecVec.size(); i++)
    {
        result[i] = vectorToPackedArray(vecVec[i]);
    }
    return result;
}

inline std::vector<std::vector<float>> arrayPackedArrayToVectorVector(const godot::TypedArray<godot::PackedFloat32Array>& arr)
{
    std::vector<std::vector<float>> result;
    result.reserve(arr.size());
    for (size_t i = 0; i < arr.size(); i++)
    {
        result.push_back(packedArrayToVector(arr[i]));
    }
    return result;
}
