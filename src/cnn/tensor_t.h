#pragma once

#include <cassert>
#include <vector>
#include <string.h>
#include <ostream>

#include "point_t.h"

// tensor is the basic data type in CNNs
template<typename T>
struct tensor_t {
  std::vector<T> data;

  tdsize size{};

  tensor_t(int _x, int _y, int _z) {
    data = std::vector<T>(static_cast<unsigned long>(_x * _y * _z));
    size.x = _x;
    size.y = _y;
    size.z = _z;
  }

  // basic overload operation
  tensor_t(const tensor_t &other) {
    data = other.data;
    size = other.size;
  }

  tensor_t<T> operator+(tensor_t<T> &other) {
    tensor_t<T> clone(*this);
    for (int i = 0; i < other.size.x * other.size.y * other.size.z; i++)
      clone.data[i] += other.data[i];
    return clone;
  }

  tensor_t<T> operator-(tensor_t<T> &other) {
    tensor_t<T> clone(*this);
    for (int i = 0; i < other.size.x * other.size.y * other.size.z; i++)
      clone.data[i] -= other.data[i];
    return clone;
  }

  T &operator()(int _x, int _y, int _z) {
    return this->get(_x, _y, _z);
  }

  T &get(int _x, int _y, int _z) {
    assert(_x >= 0 && _y >= 0 && _z >= 0);
    assert(_x < size.x && _y < size.y && _z < size.z);

    return data[
        _z * (size.x * size.y) +
            _y * (size.x) +
            _x
    ];
  }


};

