#pragma once

#include <vector>
#include <algorithm>
#include <math.h>
#include <aadc/idouble.h>
#include "Curves.h"
#include <random>
#include <nlohmann/json.hpp>
using json = nlohmann::json;
#include "DataTools.h"

////////////////////////////////////////////////////
//
//  HullWhiteZDBCurve
//
//  Contains current value of the rate and provides computation of bond price using HW formula
//
//  r_current          Current value of rate
//  sigma_0            Volatility
//  alpha              Mean reversion speed
//  time_counter       Current value of time
//  mean_reversion     Mean reversion curve
//
//  <mdouble>:     double, aadc:idouble, adept::adouble
//
////////////////////////////////////////////////////

template<class mdouble>
class HullWhiteZDBCurve 
{
public:
HullWhiteZDBCurve (
        const mdouble& r_current, 
        const mdouble& sigma_0, 
        const mdouble& alpha, 
        const double time_counter,
        const std::shared_ptr<PiecewiseLinearCurve<mdouble>>& mean_reversion
    ) 
        : m_r_current(r_current) 
        , m_sigma(sigma_0) 
        , m_alpha(alpha) 
        , m_time_counter(time_counter) 
        , m_mean_reversion(mean_reversion)
    {
        using namespace std; 
        int t_cur_index = m_mean_reversion->interpolatedIndex(m_time_counter);

        if (m_mean_reversion->getTimes().size()<2) {return;}  
        
        m_cumulat1 = std::make_shared<std::vector<mdouble>>(m_mean_reversion->getTimes().size());
        m_cumulat2 = std::make_shared<std::vector<mdouble>>(m_mean_reversion->getTimes().size());
        
        (*m_cumulat1)[0]=0;
        (*m_cumulat2)[0]=0;
        
        mdouble delta_t, mr_average;
        for(int u=0; u<m_mean_reversion->getTimes().size()-1; u++) {         // Function C(t,T) Glasserman p.114 (The first term)
            delta_t=(m_mean_reversion->getTimes()[u+1]- m_mean_reversion->getTimes()[u]);   // The order in integration is changed     
            mr_average=m_mean_reversion->getVals()[u+1]; // since m_vals[0] is responsibe for values before the first jump
            (*m_cumulat1)[u+1] = (*m_cumulat1)[u] + delta_t*mr_average;
            (*m_cumulat2)[u+1] = (*m_cumulat2)[u] - (exp(m_alpha*m_mean_reversion->getTimes()[u+1]) - 
                exp(m_alpha*m_mean_reversion->getTimes()[u]) ) * mr_average/m_alpha
            ;   
        }        
    }

    ////////////////////////////////////////////////////
    //
    //  HullWhiteZDBCurve
    //
    //  Contains current value of rate and provides computation of the bond price using HW formula
    //
    //  r_current          Current value of rate
    //  sigma_0            Volatility
    //  alpha              Mean reversion speed
    //  time_counter       Current value of time
    //  mean_reversion     Mean reversion curve
    //  cumulat1           Vectors cumulat[i] = integrals between i and i+1 nodes
    //  cumulat2           
    //
    ////////////////////////////////////////////////////

    HullWhiteZDBCurve (
        const mdouble& r_current, 
        const mdouble& sigma_0, 
        const mdouble& alpha, 
        const double time_counter,
        const std::shared_ptr<PiecewiseLinearCurve<mdouble>>& mean_reversion,
        const std::shared_ptr<std::vector<mdouble>>& cumulat1,      // Vectors cumulat[i] = integral between i and i+1 nodes
        const std::shared_ptr<std::vector<mdouble>>& cumulat2       // (HW formula for bonds)
    )  
        : m_r_current(r_current) 
        , m_sigma(sigma_0) 
        , m_alpha(alpha) 
        , m_time_counter(time_counter)
        , m_mean_reversion(mean_reversion)
        , m_cumulat1(cumulat1)
        , m_cumulat2(cumulat2)
    {}

    ~HullWhiteZDBCurve () {}

    double getCurrentTime () const {
        return m_time_counter;
    }

    // Implements HW formula
    mdouble operator () (const qtime& t) const {
        return bond(t/365.0);
    }

    // Implements HW formula
    mdouble operator () (const double& t_years) const {
        return bond(t_years);
    }

private:
    mdouble bond (const double& t_years) const {
        using namespace std;   
        mdouble s_part  = exp(- m_alpha * (t_years - m_time_counter));
        mdouble A_t_T   = (1 - s_part) / m_alpha;
        int T_index     = m_mean_reversion->interpolatedIndex(t_years); 
        int t_cur_index = m_mean_reversion->interpolatedIndex(m_time_counter);
        mdouble delta_t, integral;        
        mdouble mr_average=m_mean_reversion->getVals()[t_cur_index];
        mdouble C_t_T = m_sigma * m_sigma / (2 * m_alpha * m_alpha) * (
            t_years-m_time_counter + 1 / (2 * m_alpha) * (1 - s_part * s_part) + 2 / m_alpha * (s_part - 1)
        );

        if (T_index == t_cur_index) {   // The case when there is no entries from vectors m_cumulate1(2) is required
            mr_average = m_mean_reversion->getVals()[t_cur_index];
            integral = (t_years - m_time_counter - (1 - s_part) / m_alpha) * mr_average;   
            C_t_T -= integral;
            return exp(-A_t_T * m_r_current + C_t_T);
        }

        delta_t   = m_mean_reversion->getTimes()[t_cur_index] - m_time_counter;
        mr_average= m_mean_reversion->getVals()[t_cur_index];
        integral = (
            delta_t - (
                exp(-m_alpha * (t_years - m_mean_reversion->getTimes()[t_cur_index])) - s_part 
            ) / m_alpha
        ) * mr_average;   
        C_t_T -= integral;
        
        // Between nodes t_cur_index and T_index 
        if (T_index-t_cur_index > 1) {   // only if there are at least 2 nodes between begin and end
            C_t_T -= (*m_cumulat1)[T_index-1] - (*m_cumulat1)[t_cur_index] + (
                (*m_cumulat2)[T_index-1] - (*m_cumulat2)[t_cur_index]
            ) * exp(- m_alpha * t_years);
        }
        // between T_index-1 and T
        delta_t= t_years - m_mean_reversion->getTimes()[T_index-1];        
        mr_average=m_mean_reversion->getVals()[T_index];
        integral = (delta_t - (1 - exp(-m_alpha * delta_t)) / m_alpha) * mr_average;   
        C_t_T-=integral;
        return exp(-A_t_T * m_r_current + C_t_T);
    }

private:
    mdouble m_sigma, m_alpha;
    std::shared_ptr<PiecewiseLinearCurve<mdouble>> m_mean_reversion;
    std::vector<std::shared_ptr<PiecewiseLinearCurve<mdouble>>> m_spreads;
    mdouble m_r_current;
    double m_time_counter;
    std::shared_ptr<std::vector<mdouble>> m_cumulat1,  m_cumulat2;
};

////////////////////////////////////////////////////
//
//  ProjectCurve 
//
//  to be used for pricing of FloatLegs
//
//  discount_crv     Discount curve
//  spread_crv       Spread curve
//
//  <mdouble>:     double, aadc:idouble, adept::adouble
//
////////////////////////////////////////////////////

template<class mdouble>
class ProjectCurve {
public: 
    ProjectCurve (
        const std::shared_ptr<HullWhiteZDBCurve<mdouble>> discount_crv, 
        const std::shared_ptr<PiecewiseLinearCurve<mdouble>> spread_crv
    ) : m_discount(discount_crv) , m_project(spread_crv)
    {}
    
    ~ProjectCurve() {}

    mdouble operator () (const qtime& t) const {
        using namespace std;   
        double t_years= t/365.0;
        return (*m_discount)(t_years)*exp(-(*m_project)(t_years)*(t_years-m_discount->getCurrentTime()));
    }
private:
    std::shared_ptr<HullWhiteZDBCurve<mdouble>> m_discount;
    std::shared_ptr<PiecewiseLinearCurve<mdouble>> m_project; 
};


////////////////////////////////////////////////////
//
//  HullWhiteIrMarket
//
//  Market to price interest rate products simulated by HW model
//
//  discount_crv     Discount curve
//  project_curves   vector of project curves
//
//  <mdouble>:     double, aadc:idouble, adept::adouble
//
////////////////////////////////////////////////////

template<class mdouble>
class HullWhiteIrMarket {
public:
    typedef HullWhiteZDBCurve<mdouble> DiscountCurve; 
    typedef ProjectCurve<mdouble> ProjectCurveT;
public: 
    HullWhiteIrMarket (
        const std::shared_ptr<DiscountCurve>& discount_crv, 
        const std::vector<std::shared_ptr<ProjectCurveT>>& project_curves   
    ) : m_discount(discount_crv) , m_project_curves(project_curves)
    {}
    
    ~HullWhiteIrMarket() {}

    const DiscountCurve& getDiscountCurve() const  { 
        return *m_discount;
    }
    const ProjectCurveT& getProjectCurve(const int spread_id) const {
        return *(m_project_curves[spread_id]);
    }
private:
    std::shared_ptr<DiscountCurve> m_discount;
    std::vector<std::shared_ptr<ProjectCurveT>> m_project_curves;
};

////////////////////////////////////////////////////
//
//  HullWhiteMarketModel
// 
//  Implements Vasicek process and interface for bonds pricing
//
//  r_init             Initial interest rate 
//  sigma_0            Volatility
//  alpha              Mean Reversion speed
//  mean_reversion     Mean Reversion vector
//  spreads            Vector of spread curves
//
//  <mdouble>:     double, aadc:idouble, adept::adouble
//
////////////////////////////////////////////////////

template<class mdouble>
class HullWhiteMarketModel {
public:
    HullWhiteMarketModel(
        const mdouble& r_init, 
        const mdouble& sigma_0, 
        const mdouble& alpha, 
        const std::shared_ptr<PiecewiseLinearCurve<mdouble>>& mean_reversion,
        const std::vector<std::shared_ptr<PiecewiseLinearCurve<mdouble>>>& spreads  
    )  
        : m_r_init(r_init)
        , m_sigma(sigma_0) 
        , m_alpha(alpha) 
        , m_mean_reversion(mean_reversion)
        , m_spreads(spreads)
    {}
    
    ////////////////////////////////////////////////////
    //
    //   HullWhiteMarketModel 
    //
    //   Implements Vasicek process and interface for bonds pricing
    //
    //  json data:                Model configuration
    //  "ProjectionSpread.3M      json     parameters for HW 3month spread 
    //  "ProjectionSpread.6M      json     parameters for HW 6month spread 
    //  "ProjectionSpread.12M     json     parameters for HW 12month spread 
    //  "HWMeanReversionCurve"    json     parameters for HW mean reversion curve 
    //  "alpha"                   double   Vasicek mean rev speed 
    //  "r0"                      double   initial value of rate 
    //  "sigma"                   double   Vasicek volatility 
    //
    ////////////////////////////////////////////////////    

    HullWhiteMarketModel(const json& data) 
        : m_r_init(getParameter<mdouble>(data["r0"]))
        , m_sigma(getParameter<mdouble>(data["sigma"]))
        , m_alpha(getParameter<mdouble>(data["alpha"]))
        , m_mean_reversion(
            std::make_shared<PiecewiseLinearCurve<mdouble>>(createPWCurve<mdouble>(data["HWMeanReversionCurve"]))
        )
        , m_spreads(
            { 
                std::make_shared<PiecewiseLinearCurve<mdouble>>(createPWCurve<mdouble>(data["ProjectionSpread.3M"]))
                , std::make_shared<PiecewiseLinearCurve<mdouble>>(createPWCurve<mdouble>(data["ProjectionSpread.6M"]))
                , std::make_shared<PiecewiseLinearCurve<mdouble> >(createPWCurve<mdouble>(data["ProjectionSpread.12M"]))
            }
        ) 
    {}

    ~HullWhiteMarketModel() {}

    // Initialize process
    void initT0(const qtime& t0) {
        m_current_t=t0/365.0;
        m_time_counter=0; // time since t0
        m_r_current=m_r_init;
        m_r_accumulated=0;
    }

    // Move process state to the next point of time
    void nextQT(const mdouble& normals, const qtime& next_t) {
        nextT(normals, next_t/365.0);
    }
        
    // NextQT helper
    void nextT(const  mdouble& normals, const double& next_t) {
        using namespace std;   
        double delta_t = next_t - m_current_t;
        m_current_t = next_t;        
        mdouble mr_average, s_part, mu;
        s_part = exp(-m_alpha * delta_t);  
        
        m_time_counter += delta_t;
        //See (3.43-3.45 Glaaserman, page 110) we assume that b(t) is constant between m_current and next_t
        mr_average=(*m_mean_reversion)(m_time_counter);  
        
        mu = (1-s_part) * mr_average;   // alpha b \int_0^t e^{-alpha(t-s)}ds, b is constant 
        m_r_current = m_r_current * s_part + mu + m_sigma * normals * sqrt( (1 - s_part * s_part) / (2 * m_alpha) );
        m_r_accumulated += m_r_current;
    }

    // Returns actual IR Market
    const HullWhiteIrMarket<mdouble> getIrMarket() const {
        std::vector<std::shared_ptr<ProjectCurve<mdouble>>> project_curves;
        std::shared_ptr<HullWhiteZDBCurve<mdouble>> current_curve = getCurrentCurve();
        for (int i=0; i<m_spreads.size(); i++) {
            std::shared_ptr<ProjectCurve<mdouble>> curve = std::make_shared<ProjectCurve<mdouble>>(
                current_curve, m_spreads[i]
            );
            project_curves.push_back(curve);
        }
        return HullWhiteIrMarket<mdouble>(current_curve, project_curves);
    }

    // Return current curve 
    const std::shared_ptr<HullWhiteZDBCurve<mdouble>> getCurrentCurve() const {
        return std::make_shared<HullWhiteZDBCurve<mdouble>>(
            m_r_current, 
            m_sigma, m_alpha, 
            m_time_counter, 
            m_mean_reversion
        );
    }

    const mdouble& getSigma() const { return m_sigma; }
    const mdouble& getAlpha() const { return m_alpha; }
    const mdouble& getR0() const { return m_r_init; }
    const mdouble& getRateAccumulated() const { return m_r_accumulated; }
    const std::shared_ptr<PiecewiseLinearCurve<mdouble>>& getMeanRev() const { return m_mean_reversion; }
    const std::vector<std::shared_ptr<PiecewiseLinearCurve<mdouble>>>& getSpreads() const {
        return m_spreads;
    }
private:
    // Vasicek process input data
    const mdouble m_r_init, m_alpha, m_sigma;
    const std::shared_ptr<PiecewiseLinearCurve<mdouble>> m_mean_reversion;
    const std::vector<std::shared_ptr<PiecewiseLinearCurve<mdouble>>> m_spreads;

    mdouble m_r_current;
    mdouble m_r_accumulated;
    double m_current_t;
    double m_time_counter;
};


////////////////////////////////////////////////////
//
//  Vasicek
// 
//  Only! to be used to check the correctness of the HW-formula by MC
//
//  time        bond execution data     
//  model       HW Market Model
//  normals     vector of random variables required for simulation of one path 
//
//  <mdouble>:     double, aadc:idouble, adept::adouble
//
////////////////////////////////////////////////////

template<class mdouble>
double Vasicek (const qtime time, HullWhiteMarketModel<mdouble>& model, const std::vector<double>& normals) {
    int iterations = normals.size();
    double delta_t=double(time)/(iterations*365);
    double t=0;
    for (int i=0; i<iterations; i++) {
        t+=delta_t;
        model.nextT(normals[i],t);
    }
    return exp(-model.getRateAccumulated()*delta_t);
}