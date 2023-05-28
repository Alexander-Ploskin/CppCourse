#include <iostream>
#include <vector>
#include <algorithm>
#include <omp.h>
#include <functional>
#include <cmath>

// The map function applies a user-defined unary function f to each element of the data vector.
template<typename T, typename F>
auto map_parallel(const std::vector<T>& data, F f)
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
auto reduce_parallel(const std::vector<T>& data, F f) {
    auto result = data[0];

    #pragma omp declare reduction(my_reduce : T : omp_out+=omp_in) // declare custom reduction for f
    #pragma omp parallel for reduction(my_reduce:result) // use custom reduction in parallel loop
    for (int i = 1; i < data.size(); ++i) {
        result = f(result, data[i]); // apply f to each pair of elements in parallel
    }

    return result;
}

// The map-reduce function performs unary_function to each element of the data, and than folds the transformed data by binary_function.
// Note that binary_function should be left-associative.
// Calculations are performed parallel, so this function can be used to process large data.
template<class T, class UFunc, class BiFunc>
auto map_reduce_parallel(const std::vector<T>&data, UFunc unary_function, BiFunc binary_function) {
    return reduce_parallel(map_parallel(data, unary_function), binary_function);
}

// The map function applies a user-defined unary function f to each element of the data vector.
template<typename T, typename F>
auto map(const std::vector<T>& data, F f)
{
    std::vector<decltype(f(data[0]))> result(data.size());
    
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
    
    for (int i = 1; i < data.size(); ++i) {
        result = f(result, data[i]);
    }
    
    return result;
}

// The map-reduce function performs unary_function to each element of the data, and than folds the transformed data by binary_function.
template<class T, class UFunc, class BiFunc>
auto map_reduce(const std::vector<T>&data, UFunc unary_function, BiFunc binary_function) {
    return reduce(map(data, unary_function), binary_function);
}

// creates a vector of specified size with values from 1 to 5
auto create_test_data(size_t size) {
    std::vector<int> v(size);
    for(int i = 0; i < size; i++) {
        v[i] = i % 5 + 1;
    }
    return v;
}

int main() {
     // Unary operator for map
    auto square = [](double x){ return std::pow(x, 2); };
    
    // Binary operator for reduce
    auto add = [](double x, double y){ return x + y; };

    for (int i = 1e5; i < 1e9; i *= 10) {
        auto data = create_test_data(i);
        
        auto start_time = omp_get_wtime();
        int result = map_reduce(create_test_data(i), square, add);
        auto end_time = omp_get_wtime();
        auto elapsed_time = end_time - start_time;
        
        auto parallel_start_time = omp_get_wtime();
        int parallel_result = map_reduce_parallel(create_test_data(i), square, add);
        auto parallel_end_time = omp_get_wtime();
        auto parallel_elapsed_time = parallel_end_time - parallel_start_time;

        if (parallel_result != result) {
            std::cout << parallel_result << std::endl;
            std::cout << result << std::endl;
            throw;
        }

        std::cout << "Input size: " << i << std::endl;
        std::cout << "Elapsed time: " << elapsed_time << std::endl;
        std::cout << "Parallel elapsed time: " << parallel_elapsed_time << std::endl;
        std::cout << "Parallel is faster by " << elapsed_time * 100 / parallel_elapsed_time - 100 << " percent" << std::endl;
        std::cout << std::endl;
    }
}