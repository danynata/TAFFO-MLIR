#ifndef LIBAFFINE_HPP
#define LIBAFFINE_HPP

#include <stdint.h>

#include <algorithm>
#include <atomic>
#include <iostream>
#include <llvm/ADT/APFloat.h>
#include <sstream>
#include <vector>

namespace LibAffine {
#define MAX_ERR_SYMBOLS 1024
class Range {
public:
  llvm::APFloat start;
  llvm::APFloat end;
  Range(llvm::APFloat start, llvm::APFloat end) : start(start), end(end) {};
  llvm::APFloat get_central_value() {
    return (start + end) / (llvm::APFloat)2.0;
  };
  llvm::APFloat get_radius() { return (end - start) / (llvm::APFloat)2.0; };
};

class Var {
private:
  Var() {
    noise_symbol_coeffs.reserve(MAX_ERR_SYMBOLS);
    noise_symbol_index.reserve(MAX_ERR_SYMBOLS);
  };

  inline static std::atomic<unsigned int> highest_err_symbol{0};
  inline static std::atomic<unsigned int> inc_err_symbol_index() {
    return highest_err_symbol++;
  };
  llvm::APFloat c_value = llvm::APFloat(0.0);     // Central value
  std::vector<llvm::APFloat> noise_symbol_coeffs; // noise symbol coefficients
  std::vector<unsigned int> noise_symbol_index;   // noise symbol coefficients
  llvm::APFloat beta = llvm::APFloat(0.0);        // perturbation term
  inline llvm::APFloat
  lookup_noise_symbol_coeff(unsigned int symbol_number) const {
    for (unsigned int i = 0; i < noise_symbol_index.size(); i++) {
      if (noise_symbol_index[i] == symbol_number) {
        return noise_symbol_coeffs[i];
      }
    }
    return llvm::APFloat(0.0);
  }

public:
  Var(Range range) {
    Var();
    c_value = range.get_central_value();
    noise_symbol_coeffs.push_back(range.get_radius());
    noise_symbol_index.push_back(inc_err_symbol_index());
  };

  Range get_range() const {
    llvm::APFloat radius = abs_coeff_sum();
    return Range(c_value - radius - beta, c_value + radius + beta);
  }
  llvm::APFloat abs_coeff_sum() const {
    llvm::APFloat sum = llvm::APFloat(0.0);
    for (auto i : noise_symbol_coeffs) {
      sum = sum + llvm::abs(i);
    }
    return sum;
  }
  std::string print() const {
    std::ostringstream os;
    os << "Central Value: " << c_value.convertToDouble() << "\n";
    os << "Error Symbols and Coefficients:\n";
    for (size_t i = 0; i < noise_symbol_coeffs.size(); ++i) {
      os << "Symbol " << noise_symbol_index[i] << ": "
         << noise_symbol_coeffs[i].convertToDouble() << "\n";
    }
    // print perturbation term
    os << "Perturbation Term: " << beta.convertToDouble() << "\n";
    // print the range
    Range range = get_range();
    os << "Range: [" << range.start.convertToDouble() << ", "
       << range.end.convertToDouble() << "]\n";
    return os.str();
  }

  std::string print_affine_form() const {
    std::ostringstream os;
    os << "X = " << c_value.convertToDouble();
    for (size_t i = 0; i < noise_symbol_coeffs.size(); ++i) {
      os << " + " << noise_symbol_coeffs[i].convertToDouble() << "e"
         << noise_symbol_index[i];
    }
    // print perturbation term
    os << " + " << beta.convertToDouble() << "eu" << "\n";
    return os.str();
  }

  bool is_subset_of(const Var &b) const {
    auto a_range = get_range();
    auto b_range = b.get_range();

    // Check if the range is infinite
    if (c_value.isNaN() || b.c_value.isNaN()) {
      return true;
    }
    // Check if 'a' is a subset of 'b'
    if (a_range.start >= b_range.start && a_range.end <= b_range.end) {
      return true;
    }
    return false;
  }

  llvm::APFloat range_union_midpoint(const Var &b) const {
    auto a_range = this->get_range();
    auto b_range = b.get_range();
    auto minmax =
        std::minmax({a_range.start, a_range.end, b_range.start, b_range.end});
    // calculate midpoint
    return (minmax.first + minmax.second) / (llvm::APFloat)2.0;
  }

  // define the join operation on two affine variables
  Var join(const Var &b) const {
    Var result;
    result.c_value = range_union_midpoint(b);
    std::set_intersection(noise_symbol_index.begin(), noise_symbol_index.end(),
                          b.noise_symbol_index.begin(),
                          b.noise_symbol_index.end(),
                          std::back_inserter(result.noise_symbol_index));

    for (auto i : result.noise_symbol_index) {
      auto a_coeff = lookup_noise_symbol_coeff(i);
      auto b_coeff = b.lookup_noise_symbol_coeff(i);

      if ((a_coeff * b_coeff).isNegative()) {
        result.noise_symbol_coeffs.push_back(llvm::APFloat(0.0));
        continue;
      }

      auto abs_a_coeff = llvm::abs(a_coeff);
      auto abs_b_coeff = llvm::abs(b_coeff);
      if (abs_a_coeff < abs_b_coeff) {
        result.noise_symbol_coeffs.push_back(a_coeff);
      } else {
        result.noise_symbol_coeffs.push_back(b_coeff);
      }
    }
    // Calcluate the supremum of the union two ranges
    auto sup = llvm::maximum(get_range().end, b.get_range().end);
    // calculate purtrubation term
    result.beta = sup - result.c_value - result.abs_coeff_sum();
    return result;
  }

  Var operator+(const Var &b) const { // Addition
    Var result;
    result.c_value = c_value + b.c_value;
    std::set_union(noise_symbol_index.begin(), noise_symbol_index.end(),
                   b.noise_symbol_index.begin(), b.noise_symbol_index.end(),
                   std::back_inserter(result.noise_symbol_index));

    unsigned int index_ida = 0;
    unsigned int index_idb = 0;
    for (auto i : result.noise_symbol_index) {
      if (index_ida >= noise_symbol_index.size() ||
          noise_symbol_index[index_ida] != i) {
        result.noise_symbol_coeffs.push_back(b.noise_symbol_coeffs[index_idb]);
        index_idb++;
        continue;
      }
      if (index_idb >= b.noise_symbol_index.size() ||
          b.noise_symbol_index[index_idb] != i) {
        result.noise_symbol_coeffs.push_back(noise_symbol_coeffs[index_ida]);
        index_ida++;
        continue;
      }
      result.noise_symbol_coeffs.push_back(noise_symbol_coeffs[index_ida] +
                                           b.noise_symbol_coeffs[index_idb]);
      index_ida++;
      index_idb++;
    }

    // add perturbation terms
    result.beta = beta + b.beta;
    return result;
  }

  Var operator+(const llvm::APFloat b) const { // Addition with scalar
    Var result;
    result.c_value = c_value + b;
    for (auto i : noise_symbol_coeffs)
      result.noise_symbol_coeffs.push_back(i);

    // add perturbation term
    result.beta = beta;
    return result;
  }

  Var operator-(const Var &b) const {
    Var result;
    result.c_value = c_value - b.c_value;
    std::set_union(noise_symbol_index.begin(), noise_symbol_index.end(),
                   b.noise_symbol_index.begin(), b.noise_symbol_index.end(),
                   std::back_inserter(result.noise_symbol_index));

    unsigned int index_ida = 0;
    unsigned int index_idb = 0;
    for (auto i : result.noise_symbol_index) {
      if (index_ida >= noise_symbol_index.size() ||
          noise_symbol_index[index_ida] != i) {
        result.noise_symbol_coeffs.push_back(-b.noise_symbol_coeffs[index_idb]);
        index_idb++;
        continue;
      }
      if (index_idb >= b.noise_symbol_index.size() ||
          b.noise_symbol_index[index_idb] != i) {
        result.noise_symbol_coeffs.push_back(noise_symbol_coeffs[index_ida]);
        index_ida++;
        continue;
      }
      result.noise_symbol_coeffs.push_back(noise_symbol_coeffs[index_ida] -
                                           b.noise_symbol_coeffs[index_idb]);
      index_ida++;
      index_idb++;
    }
    // add perturbation terms
    result.beta = beta + b.beta;
    return result;
  }

  Var operator-(const llvm::APFloat b) const {
    Var result;
    result.c_value = c_value - b;
    for (auto i : noise_symbol_coeffs)
      result.noise_symbol_coeffs.push_back(i);

    result.beta = beta;
    return result;
  }

  Var operator-() const {
    Var result;
    result.c_value = -c_value;
    for (auto i : noise_symbol_coeffs)
      result.noise_symbol_coeffs.push_back(-i);

    result.beta = beta;
    return result;
  }

  Var operator*(const Var &b) const { // Approximation of multiplication range
    Var result;
    result.c_value = c_value * b.c_value;
    std::set_union(noise_symbol_index.begin(), noise_symbol_index.end(),
                   b.noise_symbol_index.begin(), b.noise_symbol_index.end(),
                   std::back_inserter(result.noise_symbol_index));

    unsigned int index_ida = 0;
    unsigned int index_idb = 0;
    for (auto i : result.noise_symbol_index) {
      if (index_ida >= noise_symbol_index.size() ||
          noise_symbol_index[index_ida] != i) {
        result.noise_symbol_coeffs.push_back(c_value *
                                             b.noise_symbol_coeffs[index_idb]);
        index_idb++;
        continue;
      }
      if (index_idb >= b.noise_symbol_index.size() ||
          b.noise_symbol_index[index_idb] != i) {
        result.noise_symbol_coeffs.push_back(b.c_value *
                                             noise_symbol_coeffs[index_ida]);
        index_ida++;
        continue;
      }
      result.noise_symbol_coeffs.push_back(
          c_value * b.noise_symbol_coeffs[index_idb] +
          b.c_value * noise_symbol_coeffs[index_ida]);
      index_ida++;
      index_idb++;
    }

    // Compute the approximation error in a new symbol
    llvm::APFloat sum_noise_mult = llvm::APFloat(0.0);
    for (auto i : noise_symbol_coeffs) {
      for (auto j : b.noise_symbol_coeffs) {
        sum_noise_mult = sum_noise_mult + llvm::abs(i * j);
      }
    }
    result.noise_symbol_index.push_back(inc_err_symbol_index());
    result.noise_symbol_coeffs.push_back(sum_noise_mult);

    // calculate perturbation term
    result.beta = (abs_coeff_sum() + c_value) * b.beta +
                  (b.abs_coeff_sum() + b.c_value) * beta + beta * b.beta;
    return result;
  }

  Var operator*(const llvm::APFloat b) const {
    Var result;
    result.c_value = c_value * b;
    for (auto i : noise_symbol_coeffs)
      result.noise_symbol_coeffs.push_back(i * b);

    result.beta = beta;
    return result;
  }

  Var operator/(const Var &b) const {
    auto a_range = this->get_range();
    auto b_range = b.get_range();

    llvm::APFloat corners[4] = {
        a_range.start / b_range.start,
        a_range.start / b_range.end,
        a_range.end / b_range.start,
        a_range.end / b_range.end,
    };

    llvm::APFloat lo = corners[0];
    llvm::APFloat hi = corners[0];
    for (int i = 1; i < 4; i++) {
      if (corners[i].compare(lo) == llvm::APFloat::cmpLessThan)
        lo = corners[i];
      if (corners[i].compare(hi) == llvm::APFloat::cmpGreaterThan)
        hi = corners[i];
    }

    return Var(Range(lo, hi));
  }

  Var operator/(const llvm::APFloat b) const {
    Var result;
    result.c_value = c_value / b;
    for (auto i : noise_symbol_coeffs)
      result.noise_symbol_coeffs.push_back(i / b);

    // calculate perturbation term
    result.beta = beta / b;
    return result;
  }

  Var operator!() const {
    Var result;
    result.c_value = c_value;
    for (auto i : noise_symbol_index)
      result.noise_symbol_index.push_back(i);
    for (auto i : noise_symbol_coeffs)
      result.noise_symbol_coeffs.push_back(i);

    // Add beta as a new symbol
    result.noise_symbol_index.push_back(inc_err_symbol_index());
    result.noise_symbol_coeffs.push_back(beta);
    return result;
  }

  // Make == operator a friend function of this class
  friend bool operator==(const Var &a, const Var &b);
};

// Implementing the case where the scalar is on the left side of the operator
extern Var operator+(const llvm::APFloat b, const Var var);

extern Var operator-(const llvm::APFloat b, const Var var);

extern Var operator*(const llvm::APFloat b, const Var var);

extern Var operator/(const llvm::APFloat b, const Var var);

extern bool operator==(const Var &a, const Var &b);

extern bool operator!=(const Var &a, const Var &b);
} // namespace LibAffine
#endif // LIBAFFINE_HPP