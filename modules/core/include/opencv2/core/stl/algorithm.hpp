// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level
// directory of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_CORE_STL_ALGORITHM_HPP
#define OPENCV_CORE_STL_ALGORITHM_HPP

#include "opencv2/core.hpp"
#include "opencv2/core/stl/pointer-tuple-replacer.hpp"
#include "opencv2/core/stl/util.hpp"
#include "opencv2/core/stl/variadic-continuous-checker.hpp"
#include <tuple>

namespace cv {
namespace experimental {

///@brief overload for forwarding a tuple and index sequence with cv iterators
/// replaced as pointers
template <typename... Args, std::size_t... Is>
auto all_of(std::tuple<Args...> tpl, index_sequence<Is...>) -> decltype(std::all_of(std::get<Is>(tpl)...)) {
  return std::all_of(std::get<Is>(tpl)...);
}

///@brief Forwarding for stl algorithm with the same name. Decides at runtime if
/// the iterators are replaced with pointers
/// or kept as cv iterators for non-contiguous matrices.
template <typename... Args> auto all_of(Args &&... args) -> decltype(std::all_of(std::forward<Args>(args)...)) {

  if (__iterators__replaceable(std::forward<Args>(args)...)) {
    return all_of(make_tpl_replaced(std::forward<Args>(args)...),
                  make_index_sequence<std::tuple_size<std::tuple<Args...>>::value>());
  } else {
    return std::all_of(std::forward<Args>(args)...);
  }
}

///@brief overload for forwarding a tuple and index sequence with cv iterators
/// replaced as pointers
template <typename... Args, std::size_t... Is>
auto any_of(std::tuple<Args...> tpl, index_sequence<Is...>) -> decltype(std::any_of(std::get<Is>(tpl)...)) {
  return std::any_of(std::get<Is>(tpl)...);
}

///@brief Forwarding for stl algorithm with the same name. Decides at runtime if
/// the iterators are replaced with pointers
/// or kept as cv iterators for non-contiguous matrices.
template <typename... Args> auto any_of(Args &&... args) -> decltype(std::any_of(std::forward<Args>(args)...)) {

  if (__iterators__replaceable(std::forward<Args>(args)...)) {
    return any_of(make_tpl_replaced(std::forward<Args>(args)...),
                  make_index_sequence<std::tuple_size<std::tuple<Args...>>::value>());
  } else {
    return std::any_of(std::forward<Args>(args)...);
  }
}

///@brief overload for forwarding a tuple and index sequence with cv iterators
/// replaced as pointers
template <typename... Args, std::size_t... Is>
auto none_of(std::tuple<Args...> tpl, index_sequence<Is...>) -> decltype(std::none_of(std::get<Is>(tpl)...)) {
  return std::none_of(std::get<Is>(tpl)...);
}

///@brief Forwarding for stl algorithm with the same name. Decides at runtime if
/// the iterators are replaced with pointers
/// or kept as cv iterators for non-contiguous matrices.
template <typename... Args> auto none_of(Args &&... args) -> decltype(std::none_of(std::forward<Args>(args)...)) {

  if (__iterators__replaceable(std::forward<Args>(args)...)) {
    return none_of(make_tpl_replaced(std::forward<Args>(args)...),
                   make_index_sequence<std::tuple_size<std::tuple<Args...>>::value>());
  } else {
    return std::none_of(std::forward<Args>(args)...);
  }
}

///@brief overload for forwarding a tuple and index sequence with cv iterators
/// replaced as pointers
template <typename... Args, std::size_t... Is>
auto count_if(std::tuple<Args...> tpl, index_sequence<Is...>) -> decltype(std::count_if(std::get<Is>(tpl)...)) {
  return std::count_if(std::get<Is>(tpl)...);
}

///@brief Forwarding for count_if stl algo. Decides at runtime if the iterators
/// are replaced with pointers
/// or kept as cv iterators for non-contiguous matrices.
template <typename... Args> auto count_if(Args &&... args) -> decltype(std::count_if(std::forward<Args>(args)...)) {

  if (__iterators__replaceable(std::forward<Args>(args)...)) {
    return count_if(make_tpl_replaced(std::forward<Args>(args)...),
                    make_index_sequence<std::tuple_size<std::tuple<Args...>>::value>());
  } else {
    return std::count_if(std::forward<Args>(args)...);
  }
}


///@brief overload for forwarding a tuple and index sequence with cv iterators
/// replaced as pointers
template <typename... Args, std::size_t... Is>
auto count(std::tuple<Args...> tpl, index_sequence<Is...>) -> decltype(std::count(std::get<Is>(tpl)...)) {
  return std::count(std::get<Is>(tpl)...);
}

///@brief Forwarding for count stl algo. Decides at runtime if the iterators
/// are replaced with pointers
/// or kept as cv iterators for non-contiguous matrices.
template <typename... Args> auto count(Args &&... args) -> decltype(std::count(std::forward<Args>(args)...)) {

  if (__iterators__replaceable(std::forward<Args>(args)...)) {
    return count(make_tpl_replaced(std::forward<Args>(args)...),
                    make_index_sequence<std::tuple_size<std::tuple<Args...>>::value>());
  } else {
    return std::count(std::forward<Args>(args)...);
  }
}

///@brief Forwarding for find stl algo. Decides at runtime if the iterators are
/// replaced with pointers
/// or kept as cv iterators for non-contiguous matrices. This is the overload
/// for when we do return an opencv iterator. This means, that we'll use pointer
/// arithmetic to get the offset and then add it to the begin iterator.
template <typename ReturnType, typename beginIt, typename... Args, std::size_t... Is,
          enable_if_t<std::is_base_of<cv::MatConstIterator, ReturnType>::value, bool> = true>
auto find(const ReturnType &, beginIt &&begin, std::tuple<Args...> tpl, index_sequence<Is...>) -> ReturnType {
  auto beginPtr = std::find(std::get<Is>(tpl)...);

  // Offsets to go for iterator
  std::ptrdiff_t diff = beginPtr - (decltype(beginPtr))begin.ptr;
  return begin + diff;
}

///@brief Forwarding for find stl algo. Decides at runtime if the iterators are
/// replaced with pointers
/// or kept as cv iterators for non-contiguous matrices. This is the overload
/// for when we don't return an opencv iterator
template <typename ReturnType, typename beginIt, typename... Args, std::size_t... Is,
          enable_if_t<!std::is_base_of<cv::MatConstIterator, ReturnType>::value, bool> = true>
auto find(const ReturnType &, beginIt &&, std::tuple<Args...> tpl, index_sequence<Is...>) -> ReturnType {
  return std::find(std::get<Is>(tpl)...);
}

///@brief Forwarding for find stl algo. Decides at runtime if the iterators are
/// replaced with pointers
/// or kept as cv iterators for non-contiguous matrices.
template <typename... Args> auto find(Args &&... args) -> decltype(std::find(std::forward<Args>(args)...)) {

  if (__iterators__replaceable(std::forward<Args>(args)...)) {
    using ReturnType = decltype(std::find(std::forward<Args>(args)...));

    // This has to be called first. Otherwise the lambda gets moved into the
    // wrong tuple
    auto tuple_replace = make_tpl_replaced(std::forward<Args>(args)...);
    auto indexSequence = make_index_sequence<std::tuple_size<std::tuple<Args...>>::value>();

    constexpr size_t val = __get_first_cv_it_index<Args...>();
    auto tpl_frwd = std::make_tuple(std::forward<Args>(args)...);

    return find(ReturnType(), std::get<val>(tpl_frwd), tuple_replace, indexSequence);
  } else {
    return std::find(std::forward<Args>(args)...);
  }
}

///@brief Forwarding for find_end stl algo. Decides at runtime if the iterators are
/// replaced with pointers
/// or kept as cv iterators for non-contiguous matrices. This is the overload
/// for when we do return an opencv iterator. This means, that we'll use pointer
/// arithmetic to get the offset and then add it to the begin iterator.
template <typename ReturnType, typename beginIt, typename... Args, std::size_t... Is,
          enable_if_t<std::is_base_of<cv::MatConstIterator, ReturnType>::value, bool> = true>
auto find_end(const ReturnType &, beginIt &&begin, std::tuple<Args...> tpl, index_sequence<Is...>) -> ReturnType {
  auto beginPtr = std::find_end(std::get<Is>(tpl)...);

  // Offsets to go for iterator
  std::ptrdiff_t diff = beginPtr - (decltype(beginPtr))begin.ptr;
  return begin + diff;
}

///@brief Forwarding for find_end stl algo. Decides at runtime if the iterators are
/// replaced with pointers
/// or kept as cv iterators for non-contiguous matrices. This is the overload
/// for when we don't return an opencv iterator
template <typename ReturnType, typename beginIt, typename... Args, std::size_t... Is,
          enable_if_t<!std::is_base_of<cv::MatConstIterator, ReturnType>::value, bool> = true>
auto find_end(const ReturnType &, beginIt &&, std::tuple<Args...> tpl, index_sequence<Is...>) -> ReturnType {
  return std::find_end(std::get<Is>(tpl)...);
}

///@brief Forwarding for find stl algo. Decides at runtime if the iterators are
/// replaced with pointers
/// or kept as cv iterators for non-contiguous matrices.
template <typename... Args> auto find_end(Args &&... args) -> decltype(std::find_end(std::forward<Args>(args)...)) {

  if (__iterators__replaceable(std::forward<Args>(args)...)) {
    using ReturnType = decltype(std::find_end(std::forward<Args>(args)...));

    // This has to be called first. Otherwise the lambda gets moved into the
    // wrong tuple
    auto tuple_replace = make_tpl_replaced(std::forward<Args>(args)...);
    auto indexSequence = make_index_sequence<std::tuple_size<std::tuple<Args...>>::value>();

    constexpr size_t val = __get_first_cv_it_index<Args...>();
    auto tpl_frwd = std::make_tuple(std::forward<Args>(args)...);

    return find_end(ReturnType(), std::get<val>(tpl_frwd), tuple_replace, indexSequence);
  } else {
    return std::find_end(std::forward<Args>(args)...);
  }
}

///@brief Forwarding for find_first_of stl algo. Decides at runtime if the iterators are
/// replaced with pointers
/// or kept as cv iterators for non-contiguous matrices. This is the overload
/// for when we do return an opencv iterator. This means, that we'll use pointer
/// arithmetic to get the offset and then add it to the begin iterator.
template <typename ReturnType, typename beginIt, typename... Args, std::size_t... Is,
          enable_if_t<std::is_base_of<cv::MatConstIterator, ReturnType>::value, bool> = true>
auto find_first_of(const ReturnType &, beginIt &&begin, std::tuple<Args...> tpl, index_sequence<Is...>) -> ReturnType {
  auto beginPtr = std::find_first_of(std::get<Is>(tpl)...);

  // Offsets to go for iterator
  std::ptrdiff_t diff = beginPtr - (decltype(beginPtr))begin.ptr;
  return begin + diff;
}

///@brief Forwarding for find_first_of stl algo. Decides at runtime if the iterators are
/// replaced with pointers
/// or kept as cv iterators for non-contiguous matrices. This is the overload
/// for when we don't return an opencv iterator
template <typename ReturnType, typename beginIt, typename... Args, std::size_t... Is,
          enable_if_t<!std::is_base_of<cv::MatConstIterator, ReturnType>::value, bool> = true>
auto find_first_of(const ReturnType &, beginIt &&, std::tuple<Args...> tpl, index_sequence<Is...>) -> ReturnType {
  return std::find_first_of(std::get<Is>(tpl)...);
}

///@brief Forwarding for find_first_of stl algo. Decides at runtime if the iterators are
/// replaced with pointers
/// or kept as cv iterators for non-contiguous matrices.
template <typename... Args> auto find_first_of(Args &&... args) -> decltype(std::find_first_of(std::forward<Args>(args)...)) {

  if (__iterators__replaceable(std::forward<Args>(args)...)) {
    using ReturnType = decltype(std::find_first_of(std::forward<Args>(args)...));

    // This has to be called first. Otherwise the lambda gets moved into the
    // wrong tuple
    auto tuple_replace = make_tpl_replaced(std::forward<Args>(args)...);
    auto indexSequence = make_index_sequence<std::tuple_size<std::tuple<Args...>>::value>();

    constexpr size_t val = __get_first_cv_it_index<Args...>();
    auto tpl_frwd = std::make_tuple(std::forward<Args>(args)...);

    return find_first_of(ReturnType(), std::get<val>(tpl_frwd), tuple_replace, indexSequence);
  } else {
    return std::find_first_of(std::forward<Args>(args)...);
  }
}


///@brief Forwarding for stl algo. Decides at runtime if the iterators are
/// replaced with pointers
/// or kept as cv iterators for non-contiguous matrices. This is the overload
/// for when we do return an opencv iterator. This means, that we'll use pointer
/// arithmetic to get the offset and then add it to the begin iterator.
template <typename ReturnType, typename beginIt, typename... Args, std::size_t... Is,
          enable_if_t<std::is_base_of<cv::MatConstIterator, ReturnType>::value, bool> = true>
auto transform_(const ReturnType &, beginIt &&begin, std::tuple<Args...> tpl, index_sequence<Is...>) -> ReturnType {
  auto beginPtr = std::transform(std::get<Is>(tpl)...);

  // Offsets to go for iterator
  std::ptrdiff_t diff = beginPtr - (decltype(beginPtr))begin.ptr;
  return begin + diff;
}

///@brief Forwarding for transform stl algo. Decides at runtime if the iterators
/// are replaced with pointers
/// or kept as cv iterators for non-contiguous matrices. This is the overload
/// for when we don't return an opencv iterator
template <typename ReturnType, typename beginIt, typename... Args, std::size_t... Is,
          enable_if_t<!std::is_base_of<cv::MatConstIterator, ReturnType>::value, bool> = true>
auto transform_(const ReturnType &, beginIt &&, std::tuple<Args...> tpl, index_sequence<Is...>) -> ReturnType {
  return std::transform(std::get<Is>(tpl)...);
}

///@brief Forwarding for transform stl algo. Decides at runtime if the iterators
/// are replaced with pointers
/// or kept as cv iterators for non-contiguous matrices.
template <typename... Args> auto transform(Args &&... args) -> decltype(std::transform(std::forward<Args>(args)...)) {

  if (__iterators__replaceable(std::forward<Args>(args)...)) {
    using ReturnType = decltype(std::transform(std::forward<Args>(args)...));
    // This has to be called first. Otherwise the lambda gets moved into the
    // wrong tuple
    auto tuple_replace = make_tpl_replaced(std::forward<Args>(args)...);
    auto indexSequence = make_index_sequence<std::tuple_size<std::tuple<Args...>>::value>();

    // Get second last index for calculating the correct return valuey
    constexpr size_t val = __get_second_last_index<Args...>();
    auto tpl_frwd = std::make_tuple(std::forward<Args>(args)...);

    return transform_(ReturnType(), std::get<val>(tpl_frwd), tuple_replace, indexSequence);
  } else {
    return std::transform(std::forward<Args>(args)...);
  }
}

} // namespace experimental
} // namespace cv
#endif // OPENCV_CORE_STL_ALGORITHM_HPP
