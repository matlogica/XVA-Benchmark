#pragma once

#include <chrono>
#include <algorithm>
#include <fstream>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <math.h>
#include <unordered_map>
#include <cstdlib>
#include <string>

#include <nlohmann/json.hpp>

#include <aadc/aadc_types.h>

#include "HullWhite.h"
#include "CSARules.h"
#include "SwapLegs.h"
#include "DataTools.h"

using json = nlohmann::json;

////////////////////////////////////////////////////
//
//   XVAProblem
//
//   Evaluates portfolio at future time steps and computes XVA measures
//
//   <mdouble>:     double, aadc:idouble, adept::adouble
//
////////////////////////////////////////////////////

template<typename mdouble>
class XVAProblem {
    typedef typename aadc::TypeTraits<mdouble>::mbool mbool;
public:
    XVAProblem() {}
    ~XVAProblem() {}

    ////////////////////////////////////////////////////
    //
    //   initModel 
    //
    //   initialize Market Model
    //
    //   data     Model configuration
    //
    ////////////////////////////////////////////////////    
    
    void initModel(const json& data) {
        m_model = std::make_shared<HullWhiteMarketModel<mdouble>>(data);
    }

    ////////////////////////////////////////////////////
    //
    //  initData
    //
    //  initialize XVA problem
    //
    //  json data:                          XVA task configuration
    //  "t0"                        int     initial time
    //  "Portfolio"                 json    Portfolio definition
    //  "ModelAndPricingTimes"      json    Market model and pricing times discretization
    //  "CompanySurvivalCurve"      json    Company survival curve
    //  "CounterPartySurvivalCurve" json    Counter party survival curve
    //  "csa"                       json    CSA rules
    //  "Currencies/EUR/"           json    Market model parameters    
    //
    ////////////////////////////////////////////////////

    void initData(const json& data) 
    {
        m_t0=data["t0"].get<int>();
        readPortfolio<mdouble>(m_fixed_leg, m_float_leg, data["Portfolio"], m_t0);
        readModelAndPricingTimes(m_model_times, m_is_pricing, m_pricing_times, data["ModelAndPricingTimes"], m_t0);
        
        m_company_survival_curve= std::make_shared<LinearInterpDiscountCurve<mdouble>>(
            createDiscountCurve<mdouble>(data["CompanySurvivalCurve"], m_t0)
        );
        m_ctrparty_survival_curve= std::make_shared<LinearInterpDiscountCurve<mdouble>>(
            createDiscountCurve<mdouble>(data["CounterPartySurvivalCurve"], m_t0)
        );

        m_CSA_object= std::make_shared<CSARules<mdouble>>(data["csa"]);
        initModel(data["Currencies"]["EUR"]);
    }
    
    void prepareSimulations() {
        m_NEE.resize(m_pricing_times.size());
        std::fill(m_NEE.begin(), m_NEE.end(), 0.);
        
        m_PEE.resize(m_pricing_times.size());
        std::fill(m_PEE.begin(), m_PEE.end(), 0.);
        
        m_CVA=m_DVA=0;
    }
    
    ////////////////////////////////////////////////////
    //
    //   simulatePath 
    //
    //   initialize underlying process model
    //
    //   random_vec     Vector of random number necessary for simulation of one MC path
    //
    ////////////////////////////////////////////////////

    void simulatePath(const std::vector<mdouble>& random_vec) {
        using namespace std;
        m_model->initT0(m_model_times[0]); 

        HullWhiteIrMarket<mdouble> market=m_model->getIrMarket();
        std::size_t numTrades=m_fixed_leg.size();
        int price_time_index=0;
        m_CSA_object->initCollateral();
           
        for (qtime i=0; i<m_model_times.size(); i++) {
            qtime t = m_model_times[i];
            if (i>0) m_model->nextQT(random_vec[i], t);
            if (m_is_pricing[i]) {
                market=m_model->getIrMarket();
                for (std::size_t ti = 0; ti < numTrades; ++ti) {
                    m_fixed_leg[ti]->nextT(t);
                    m_float_leg[ti]->nextT(market,t);    
                } 
                m_total_price=0;
                for (std::size_t ti = 0; ti < numTrades; ++ti) {
                    m_total_price+= m_fixed_leg[ti]->getPrice(market, t);
                    m_total_price+= m_float_leg[ti]->getPrice(market, t);
                }
                m_CSA_object->next(m_total_price);  
                                                                  
                mdouble CSA_total_price = m_total_price - m_CSA_object->getCollateral();
                
                m_PEE[price_time_index] += max(CSA_total_price, 0.);
                m_NEE[price_time_index] += min(0., CSA_total_price); 
                price_time_index++;
            }
        }
    }

    ////////////////////////////////////////////////////
    //
    //   computeXVAMeasures 
    //
    //   Compute CVA and DVA integrals
    //
    ////////////////////////////////////////////////////

    void computeXVAMeasures() {
        //Trapezoidal rule
        int i_end=m_pricing_times.size();
        const LinearInterpDiscountCurve<mdouble>& company = *m_company_survival_curve;
        const LinearInterpDiscountCurve<mdouble>& ctrparty= *m_ctrparty_survival_curve;
        mdouble ctrparty_next=ctrparty(m_pricing_times[0]);
        mdouble company_next=company(m_pricing_times[0]);
         
        for (int i = 0; i< i_end-1; i++) {
            mdouble ctrparty_prev=ctrparty_next;
            ctrparty_next=ctrparty(m_pricing_times[i+1]);
            mdouble company_prev=company_next;
            company_next=company(m_pricing_times[i+1]);

            m_CVA += (m_PEE[i]+m_PEE[i+1]) * (ctrparty_prev-ctrparty_next)*0.5;
            m_DVA += (m_NEE[i]+m_NEE[i+1]) * (company_prev-company_next)*0.5;
        }
    }

    // Returns amount of random variables necessary for simulation of one MC path
    int numberOfRandomVars() { return m_model_times.size(); }

    const std::shared_ptr<CSARules<mdouble>>& getCSA() const { return m_CSA_object; }
    const std::shared_ptr<HullWhiteMarketModel<mdouble>>& getModel() const { return m_model; }
    const std::vector<mdouble>& getPEE() const { return m_PEE; }
    const std::vector<mdouble>& getNEE() const { return m_NEE; }
    const mdouble& getDVA() const { return m_DVA; }
    const mdouble& getCVA() const { return m_CVA; }
    const qtime& getT0() const { return m_t0; }

    const std::shared_ptr<LinearInterpDiscountCurve<mdouble>>& getCompSurvCurv() const {
        return m_company_survival_curve;
    }
    const std::shared_ptr<LinearInterpDiscountCurve<mdouble>>& getCtrpSurvCurv() const {
        return m_ctrparty_survival_curve;
    }
    
public:
    json m_xva_data;
private:
    qtime m_t0;
    std::vector<std::shared_ptr<FixedLeg<mdouble>>> m_fixed_leg;
    std::vector<std::shared_ptr<FloatLeg<mdouble>>> m_float_leg;

    //simulation objects
    std::shared_ptr<HullWhiteMarketModel<mdouble>> m_model;    
    std::vector<qtime> m_model_times;
    std::vector<qtime> m_pricing_times;
    std::vector<bool> m_is_pricing;
    
    // xVA simulation
    std::shared_ptr<CSARules<mdouble>> m_CSA_object;
    mdouble m_total_price;
    std::vector<mdouble> m_PEE;
    std::vector<mdouble> m_NEE;
    std::shared_ptr<LinearInterpDiscountCurve<mdouble>> m_company_survival_curve;
    std::shared_ptr<LinearInterpDiscountCurve<mdouble>> m_ctrparty_survival_curve;

    mdouble m_DVA;
    mdouble m_CVA;
};
