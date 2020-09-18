#pragma once

#include "adept/adept.h"

inline adept::adouble iIf(const bool cond, const adept::adouble a, const double b) {
    return cond ? a : b;
}
inline adept::adouble iIf(const bool cond, const double a, const  adept::adouble b) {
    return cond ? a : b;
}
