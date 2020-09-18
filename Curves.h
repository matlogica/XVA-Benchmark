#pragma once

#include <vector>
#include <aadc/idouble.h>
#include <math.h>

typedef int qtime;
typedef double qval;

inline double qYearFrac(const qtime& start, const qtime& end) {
    return double(end - start) / 365.0;
}

////////////////////////////////////////////////////
//
//  PiecewiseLinearCurve 
//
//  operator () returns classic interpolation of values in points of interpolation
//
//  t     vector of interpolation points 
//  v     vector of values in the interpolation points
//
//  <mdouble>:     double, aadc:idouble, adept::adouble
//
////////////////////////////////////////////////////

template<typename mdouble>
class PiecewiseLinearCurve {
public:
    PiecewiseLinearCurve(const std::vector<double>& t, const std::vector<mdouble>& v)
        : m_t(t), m_vals(v)
    {}
    PiecewiseLinearCurve(const std::vector<qtime>& t, const std::vector<mdouble>& v)
        : m_vals(v) 
    {
        for(int i=0; i<t.size(); i++) m_t.push_back(t[i]/365.0);
    }
    PiecewiseLinearCurve(const PiecewiseLinearCurve<mdouble>& curve)
        : m_vals(curve.m_vals)
        , m_t(curve.m_t)
    {}
public:
    mdouble operator() (const double t) const {
        auto lb = std::lower_bound(m_t.begin(), m_t.end(), t);
        if (lb == m_t.begin()) {
            return m_vals.front();
        }
        else if (lb == m_t.end()) {
            return m_vals.back();
        }
        int64_t indx(lb - m_t.begin());
        double len_t(*lb - *(lb-1));
        double wl((*lb - t) / len_t), wr((t - *(lb-1)) / len_t);

        return wl * m_vals[indx-1] + wr * m_vals[indx];
    }

    int interpolatedIndex (const double& t) {
        auto i_t = std::lower_bound(m_t.begin(), m_t.end(), t);
        int index_t = std::distance(m_t.begin(), i_t);
        if (index_t==m_t.size()) index_t--; 
        return index_t;
    }

    const std::vector<mdouble>& getVals() const { return m_vals;}
    const std::vector<double>& getTimes() const { return m_t;}

private:
    const std::vector<mdouble> m_vals;
    std::vector<double> m_t;  // const is deleted in order to provide constructor with vector<qtime> type 
};

////////////////////////////////////////////////////
//
//  FlatDiscountCurve 
//
//  Operator (t) returns  exp(-zero_rate(t-t0))
//
//  zero_rate     
//  t0
//
//  <mdouble>:     double, aadc:idouble, adept::adouble
//
////////////////////////////////////////////////////

template<typename mdouble>
class FlatDiscountCurve {
public:
    FlatDiscountCurve(const mdouble& zero_rate, const qtime& t0)
            : mZeroRate(zero_rate), mT0(t0)
    {}
    FlatDiscountCurve(const FlatDiscountCurve& other)
            : mZeroRate(other.mZeroRate), mT0(other.mT0)
    {}
public:
    mdouble operator ()(const qtime& t) const {
        using namespace std; // for simultaneous use of AADC and ADEPT
        return exp(-mZeroRate * qYearFrac(mT0, t));
    }
private:
    const mdouble mZeroRate;
    const qtime mT0;
};

////////////////////////////////////////////////////
//
//  LinearInterpDiscountCurve 
//
//  operator (t) returns exp(- (t-t0) PiecewiseLinearCurve(t))
//
//  zero_rate      Zero Rate 
//  tenors         Vector of time interpolation points
//  t0             initial time
//
//  <mdouble>:     double, aadc:idouble, adept::adouble
//
////////////////////////////////////////////////////

template<typename mdouble>
class LinearInterpDiscountCurve {
public:
    LinearInterpDiscountCurve(const std::vector<mdouble>& zero_rate, const std::vector<qtime>& tenors, const qtime& t0)
        : curve(PiecewiseLinearCurve<mdouble>(tenors, zero_rate)), mT0(t0)
    {}
    LinearInterpDiscountCurve(const PiecewiseLinearCurve<mdouble>& pw_curve, const qtime& t0)
        : curve(pw_curve), mT0(t0)
    {}
    LinearInterpDiscountCurve(const LinearInterpDiscountCurve& other)
        : curve(other.curve), mT0(other.mT0)
    {}

public:
    mdouble operator()(const qtime& t) const {
        using namespace std;
        return exp(- curve(t/365.0)* qYearFrac(mT0, t));
    }
    
    const std::vector<mdouble>& getZeroRatesVector() const { return curve.getVals(); }

public:
    PiecewiseLinearCurve<mdouble> curve;
    const qtime mT0;
};
