#include <iostream>
#include <algorithm>
#include <vector>
#include <cmath>
#include <omp.h>
#include <functional>

// The map function applies a user-defined unary function f to each element of the data vector.
template<typename T, typename F>
auto map(const std::vector<T>& data, F f)
{
    std::vector<decltype(f(data[0]))> result(data.size());
    
    #pragma omp parallel for
    for (int i = 0; i < data.size(); ++i) {
        result[i] = f(data[i]);
    }
    
    return result;
}

// The reduce function takes a user-defined binary associative function f and calculates the sum of all elements.
template<typename T, typename F>
auto reduce(const std::vector<T>& data, F f)
{
    auto result = data[0];
    
    #pragma omp parallel for reduction(f:+)
    for (int i = 1; i < data.size(); ++i) {
        result = f(result, data[i]);
    }
    
    return result;
}

// The map-reduce function performs unary_function to each element of the data, and than folds the transformed data by binary_function.
// Note that binary_function should be left-associative.
// Calculations are performed parallel, so this function can be used to process large data.
template<class T, class UFunc, class BiFunc>
auto map_reduce(const std::vector<T>&data, UFunc unary_function, BiFunc binary_function) {
    return reduce(map(data, unary_function), binary_function);
}

// Usage example
int main()
{
    std::vector<double> data{1.0, 2.0, 3.0, 4.0, 5.0};
    
    std::cout << "map-reduce usage example" << std::endl;
    std::cout << "data vector: ";
    for (auto x : data) {
        std::cout << x << " ";
    }
    std::cout << std::endl;
    
    // Unary operator for map
    auto square = [](double x){ return std::pow(x, 2); };
    
    // Binary operator for reduce
    auto add = [](double x, double y){ return x + y; };

    auto norm = std::sqrt(map_reduce(data, square, add));
    
    std::cout << "calculated norm of the vector: " << norm << std::endl;
    
    return 0;
}