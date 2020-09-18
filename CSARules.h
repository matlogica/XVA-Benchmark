#pragma once
#include <aadc/ibool.h>
#include "adept_local.h"
#include <nlohmann/json.hpp>
using json = nlohmann::json;

////////////////////////////////////////////////////
//
//  CSA Rules
//
//  _th       Threshold high
//  _tl       Threshold low
//  _mta_h    Minimum transfer amount high
//  _mta_l    Minimum transfer amount low    
//  _c_t      Initial collateral 
//
//  <mdouble>:     double, aadc:idouble, adept::adouble
//
////////////////////////////////////////////////////

template<typename mdouble>
class CSARules {
typedef typename aadc::TypeTraits<mdouble>::mbool mbool;
public:
    CSARules(
        const double _th,     
        const double _tl,     
        const double _mta_h,  
        const double _mta_l,      
        const double _c_t    
    ) : 
        th(_th), tl(_tl), mta_h(_mta_h), mta_l(_mta_l), init_c_t(_c_t)
    {}
    
    CSARules(const json& data) :  
        th(data["th"].get<double>())
        , tl(data["tl"].get<double>())
        , mta_h(data["mta_h"].get<double>())
        , mta_l(data["mta_l"].get<double>())
        , init_c_t(data["c_t"].get<double>())
    {}

    ~CSARules () {}
    
    void next (mdouble p_tp1) {
        using namespace std;
        mdouble margin_call_high = max(p_tp1 - c_t - th, 0.);
        mbool high_margin_call_too_small = margin_call_high < mta_h;

        mdouble margin_call_low = min(p_tp1 - c_t - tl, 0.);
        mbool low_margin_call_too_small = margin_call_low < -mta_l;

        c_t += iIf(low_margin_call_too_small, margin_call_low, 0.)
            + iIf( high_margin_call_too_small, 0., margin_call_high)
        ;
    }

    void initCollateral () { c_t=init_c_t; }
    mdouble&  getCollateral () { return c_t; }
private:
    const double th;     
    const double tl;     
    const double mta_h;  
    const double mta_l;  
    const double init_c_t;  
    mdouble c_t;    
};
