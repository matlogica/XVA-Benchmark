#pragma once

#include <nlohmann/json.hpp>
#include <aadc/aadc.h>
#include "SwapLegs.h"

using json = nlohmann::json;

////////////////////////////////////////////////////
//
//  getArgumentsMap() 
//
//  Stores a map which connects json paths=name_of_variables, 
//  its values and corresponding AADC-indices.
//
//  <mdouble>:     double, aadc:idouble, adept::adouble
//
////////////////////////////////////////////////////

typedef std::map<std::string, std::pair<idouble, aadc::AADCArgument>> ArgumentsMap;

inline ArgumentsMap& getArgumentsMap() {
    static ArgumentsMap request_variable_inputs;
    return request_variable_inputs;
}

////////////////////////////////////////////////////
//
//  getParameter(const json& data)
//
//  Helper function to extract values from input json data.
//  This function are called when XVA's data are loaded. For a <double> version it returns a corresponding value from
//  the json. In <idouble> version getParameter returns the same idouble variable, which is associated 
//  with the value from the json.
//  it is important to save the inheritance, i.e. exactly the variable marked (and not its copy) should be used in further
//
//  data     XVA task data
//
//  <mdouble>:     double, aadc:idouble, adept::adouble
//
////////////////////////////////////////////////////

template<typename mdouble>
inline mdouble getParameter(const json& data) {
    return data.get<double>();
}

template<>
inline idouble getParameter<idouble>(const json& data) {
    if (data.is_number()) {
        return data.get<double>();
    }
    return getArgumentsMap()[data.get<std::string>()].first;
}

////////////////////////////////////////////////////
//
//  createPWCurve() 
//
//  Creates PiecewiseLinearCurve using json representation.
//
//  json data:
//  "T"          double   Curve expiry in years
//  "step"       double   Time discretization step
//  "flat_rate"  double   Flat rate for the curve
//  "bump_index" int      If present, apply bump to bucket
//  "bump_size"  double   Bump size
//
//   <mdouble>:     double, aadc:idouble, adept::adouble
//
////////////////////////////////////////////////////

template<typename mdouble>
inline PiecewiseLinearCurve<mdouble> createPWCurve(const json& data) {
    std::vector<double> times;
    std::vector<mdouble> values;
    double step = data["step"].get<double>();
    double max_t = data["T"].get<double>();
    mdouble flat_rate = getParameter<mdouble>(data["flat_rate"]);
    double t = 0;
    
    while (t < max_t) {
      values.push_back(flat_rate);
      times.push_back(t);
      t+= step;
    }
    if (data.contains("bump_index")) {
        int index(data["bump_index"].get<int>());
        double bump_size=data["bump_size"].get<double>();
        if (index < values.size()) values[index] += bump_size;
    }
    
    return PiecewiseLinearCurve<mdouble>(times, values);
}

////////////////////////////////////////////////////
//
//  createDiscountCurve(const json& data, const qtime t0)
//
//  Creates LinearInterpDiscountCurve (company_survival_curve, ctrpary_survival_curve) using json data
//
//  json data:
//  "step"        int       step of discretization 
//  "T"           int       Maximal time 
//  "flat_rate"   double    flat rate
//  t0                      initial time
//
//  <mdouble>:     double, aadc:idouble, adept::adouble
//
////////////////////////////////////////////////////

template<typename mdouble>
inline LinearInterpDiscountCurve<mdouble> createDiscountCurve(const json& data, const qtime t0) {
    std::vector<qtime> times;
    std::vector<mdouble> values;
    qtime step = data["step"].get<int>();
    qtime max_t = data["T"].get<int>();
    mdouble rate = getParameter<mdouble>(data["flat_rate"]);
    qtime t = t0;
    while (t < max_t) {
      times.push_back(t);
      values.push_back(rate);
      t+= step;
    }
    if (data.contains("bump_index")) {
        int index(data["bump_index"].get<int>());
        double bump_size=data["bump_size"].get<double>();
        if (index < values.size()) values[index] += bump_size;
    }

    return LinearInterpDiscountCurve<mdouble>(values, times, t0);
}

////////////////////////////////////////////////////
//
//  readModelAndPricingTimes(
//      std::vector<int>& model_times, std::vector<bool>& is_pricing,
//      std::vector<int>& pricing_times, const json& data, const qtime t0
//  )
//
//  Fill vectors model_times, is_pricing and pricing_times
//
//  model_times          process time discretization points 
//  is_pricing           indicates if process time is a pricing point
//  pricing_times        times where portfolio should be priced
//  t0                   initial time
//  json data:           parameters to create synthetic data
//  "T"            int   maximal time
//  "step"         int   step of discretization
//  "PricingFreq"  int   interval between pricing times
//
////////////////////////////////////////////////////

inline void readModelAndPricingTimes(
    std::vector<int>& model_times,
    std::vector<bool>& is_pricing,
    std::vector<int>& pricing_times,
    const json& data,
    const qtime t0
) {
    int max_t(data["T"].get<int>()); 
    int model_step(data["step"].get<int>()); 
    int pricing_freq(data["PricingFreq"].get<int>()); 
    int pr(0);
    int t=t0;
    while (t < max_t ) {
        model_times.push_back(t);
        if (pr == 0  || (t + model_step > max_t)) {
            pricing_times.push_back(t);
            pr = pricing_freq;
            is_pricing.push_back(true);
        } else {
            is_pricing.push_back(false);
        }
        --pr;
        t += model_step;
    }
}

////////////////////////////////////////////////////
//
//  generateFloatLeg(const qtime t0, const int& numPeriods, std::mt19937_64& gen)
//
//  Creates FloatLeg with synthetic set of cashflows 
//
//  t0           initial time 
//  numPeriods   number of cash flows
//  gen          random numbers generator
//
//  <mdouble>:     double, aadc:idouble, adept::adouble
//
////////////////////////////////////////////////////

template<class mdouble>
inline std::shared_ptr<FloatLeg<mdouble>>  generateFloatLeg(const qtime t0, const int& numPeriods, std::mt19937_64& gen) {
    std::vector<double> notionals;
    std::vector<qtime> start_times;
    std::vector<qtime> end_times;
    std::uniform_real_distribution<> uniform_distrib(0, 1);
    std::uniform_int_distribution<int> unif_int_distrib(0, 2);

    qtime start(t0+int(10 * 365 * uniform_distrib(gen)));
    int period(int(2 * 365 * uniform_distrib(gen) + 30));
    for (int ti = 0; ti < numPeriods; ++ti) {
        start_times.push_back(start); start += period;
        end_times.push_back(start);
        notionals.push_back(2*uniform_distrib(gen) - 0.985);
    }
    return std::make_shared<FloatLeg<mdouble>>(notionals, start_times, end_times, end_times, unif_int_distrib(gen));
}

////////////////////////////////////////////////////
//
//  generateFixedLeg(const qtime t0, const int& numPeriods, std::mt19937_64& gen)
//
//  Creates FixedLeg with synthetic set of cashflows 
//
//  t0           initial time 
//  numPeriods   number of cash flows
//  gen          random numbers generator
//
//  <mdouble>:     double, aadc:idouble, adept::adouble
//
////////////////////////////////////////////////////

template<class mdouble>
inline std::shared_ptr<FixedLeg<mdouble>> generateFixedLeg(const qtime t0, const int& numPeriods, std::mt19937_64& gen) {
    std::vector<double> notionals;
    std::vector<qtime> start_times;
    std::vector<qtime> end_times;
    std::uniform_real_distribution<> uniform_distrib(0, 1);

    qtime start(t0+int(10 * 365 * uniform_distrib(gen)));
    int period(int(2 * 365 * uniform_distrib(gen) + 30));
    for (int ti = 0; ti < numPeriods; ++ti) {
        start_times.push_back(start); start += period;
        end_times.push_back(start);
        notionals.push_back(0.15 * (uniform_distrib(gen) * 2.0 - 1.0));
    }
    return std::make_shared<FixedLeg<mdouble>>(notionals, end_times);
}

////////////////////////////////////////////////////
//
//  readPortfolio(
//      std::vector<std::shared_ptr<FixedLeg<mdouble>>>& fixed_legs,
//      std::vector<std::shared_ptr<FloatLeg<mdouble>>>& float_legs,
//      const json& portfolio, qtime t0
//  )
//
//  Fill vectors of FixedLeg and FloatLeg 
//
//  fixed_legs             vector of fixed legs 
//  float_legs             vector of float legs 
//  json portfolio data:
//  "NumRandomTrades"      int  Number of legs
//  "NumPeriods"           int  Number of cash flows for each leg
//  t0                     initial time
//
//  <mdouble>:     double, aadc:idouble, adept::adouble
//
////////////////////////////////////////////////////

template<typename mdouble>
inline void readPortfolio(
    std::vector<std::shared_ptr<FixedLeg<mdouble>>>& fixed_legs,
    std::vector<std::shared_ptr<FloatLeg<mdouble>>>& float_legs,
    const json& portfolio,
    const qtime t0
) {
    std::mt19937_64 gen(17);
    int num_trades(portfolio["NumRandomTrades"].get<int>());
    int num_periods(portfolio["NumPeriods"].get<int>());
    for (int ti = 0; ti < num_trades; ++ti) {
        fixed_legs.push_back(generateFixedLeg<mdouble>(t0, num_periods, gen));
        float_legs.push_back(generateFloatLeg<mdouble>(t0, num_periods, gen));
    }
}

////////////////////////////////////////////////////
//
// generateRandomCurve(qtime t0, std::vector<qtime>& tenors, std::mt19937_64& gen)
//
//  Creates LinearInterpDiscountCurve  
//
//  t0        initial time 
//  tenors    interpolation points
//  gen       random numbers generator
//
//  <mdouble>:     double, aadc:idouble, adept::adouble
//
////////////////////////////////////////////////////

inline LinearInterpDiscountCurve<mdouble> generateRandomCurve(const qtime t0, const std::vector<qtime>& tenors, std::mt19937_64& gen) {
    std::uniform_real_distribution<> uniform_distrib(0, 1);
    std::vector<mdouble> zr(tenors.size());
    double db = 1.0;
    for (int i = 0; i < zr.size(); ++i) {
        double fwd = uniform_distrib(gen) * 0.15;
        db *= std::exp(-fwd * qYearFrac(i > 0 ? tenors[i - 1] : t0, tenors[i]));
        zr[i] = -log(db) / qYearFrac(t0, tenors[i]);
    }
    return LinearInterpDiscountCurve<mdouble>(zr, tenors, t0);
}