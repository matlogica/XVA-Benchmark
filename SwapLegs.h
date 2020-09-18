#pragma once
#include <vector>
#include <iostream>
#include <memory>
#include <random>
#include "Curves.h"

////////////////////////////////////////////////////
//
//  FixedLeg
//
//  amounts    Values of the Cash Flows
//  times      Payment dates
//
//  <mdouble>:     double, aadc:idouble, adept::adouble
//
////////////////////////////////////////////////////

template<typename mdouble>
class FixedLeg {
public:
    FixedLeg (
        const std::vector<double>& amounts, const std::vector<qtime>& times
    ) 
        : mAmounts(amounts), mTimes(times)
    {}
public:
    // Implements Leg initialization  
    template<class Market>
    void initT0 (const Market& m0, const qtime& t0) {
        mT0FirstIndx = mFirstCfIndx = moveCFIndx(0, t0);
    }

    // Make the Leg state  in concordance with the market state
    void nextT (const qtime& next_t) {
        mFirstCfIndx = moveCFIndx(mFirstCfIndx, next_t);
    }

    // Returns price of the FixedLeg
    template<class Market>
    mdouble getPrice(const Market& m, const qtime& t) const {
        mdouble price(0.);
        const typename Market::DiscountCurve& curve=m.getDiscountCurve();
        for (int i = mFirstCfIndx; i < mTimes.size(); ++i) {
		if (idouble::recording) CAAD_LoopPulse(i - mFirstCfIndx);
            price = price + mAmounts[i] * curve(mTimes[i]);
        }
        return price;
    }

    template<class Market>
    mdouble getDeltaPrice (const Market& delta_m, const qtime& t) const {
        return getPrice(delta_m, t);
    }

    void resetT0 () {
        mFirstCfIndx = mT0FirstIndx;
    }
private:
    int moveCFIndx (const int& curr_indx, const qtime& time) {
        int indx(curr_indx);
        for (indx = 0; indx < mTimes.size() && mTimes[indx] < time; ++indx) {}
        return indx;
    }

private:
    const std::vector<double> mAmounts;
    const std::vector<qtime> mTimes;
    int mFirstCfIndx, mT0FirstIndx;
};

////////////////////////////////////////////////////
//
//  FloatLeg
//
//  notionals       Values of the Cash Flows
//  start_times     
//  end_times
//  pay_times       Payment times
//  spread_id       Spread curve ID
//
//  <mdouble>:     double, aadc:idouble, adept::adouble
//
////////////////////////////////////////////////////

template<typename mdouble>
class FloatLeg {
public:
    FloatLeg (
        const std::vector<double>& notionals
        , const std::vector<qtime>& start_times
        , const std::vector<qtime>& end_times
        , const std::vector<qtime>& pay_times
        , const int& spread_id
    )
        : mNotionals(notionals)
        , mStartTimes(start_times)
        , mEndTimes(end_times)
        , mPayTimes(pay_times)
        , mFwds(mPayTimes.size(), 0.)
        , m_spread_id(spread_id)
    {}

public:
    // Implements Leg initialization  
    template<class Market>
    void initT0 (const Market& m0, const qtime& t0) {
        mT0FirstIndx = mFirstCfIndx = moveCFIndx(0, t0);
    }

    // Make the Leg state  in concordance with the market state
    template<class Market>
    void nextT (const Market& next_m, const qtime& next_t) {
        mFirstCfIndx = moveCFIndx(mFirstCfIndx, next_t);
    }

    // Returns price of the FloatLeg
    template<class Market>
    mdouble getPrice (const Market& m, const qtime& t) {
        mdouble price(0.);
        const typename Market::DiscountCurve& curve=m.getDiscountCurve();
        const typename Market::ProjectCurveT& proj_curve=m.getProjectCurve(m_spread_id); 
        for (int i = mFirstCfIndx; i < mPayTimes.size(); ++i) {
            if (idouble::recording) CAAD_LoopPulse(i - mFirstCfIndx);
            mdouble fwd;
            if (mStartTimes[i] >= t) {
                fwd = (proj_curve(mStartTimes[i]) / proj_curve(mEndTimes[i]) - 1.0)
                    / qYearFrac(mStartTimes[i], mEndTimes[i])
                ;
                mFwds[i] = fwd; // store fwd rates for future pricing
            } else {
                fwd = mFwds[i]; // use previously forcasted rate. TODO: Change to fixing framework
            }
            price = price + mNotionals[i] * fwd * curve(mPayTimes[i]);
        }
        return price;
    }

    void resetT0 () {
        mFirstCfIndx = mT0FirstIndx;
    }

private:
    int moveCFIndx (const int& curr_indx, const qtime& time) {
        int indx(curr_indx);
        for (indx = 0; indx < mPayTimes.size() && mPayTimes[indx] < time; ++indx) {}
        return indx;
    }

private:
    const std::vector<double> mNotionals;
    const std::vector<qtime> mStartTimes, mEndTimes, mPayTimes;
    std::vector<mdouble> mFwds;
    int mFirstCfIndx, mT0FirstIndx;
    int m_spread_id;
};

////////////////////////////////////////////////////
//
//  IRMarket
//
//  discount     Discount curve
//  proj         Projected curve
//
//  <mdouble>:     double, aadc:idouble, adept::adouble
//
////////////////////////////////////////////////////

template<typename mdouble>
class IRMarket {
public:
    typedef LinearInterpDiscountCurve<mdouble> DiscountCurve;
    typedef LinearInterpDiscountCurve<mdouble> ProjectCurveT;
public:
    IRMarket (
        const LinearInterpDiscountCurve<mdouble>& discount, const LinearInterpDiscountCurve<mdouble>& proj
    )
        : m_discount(discount), m_proj(proj)
    {}
    const DiscountCurve& getDiscountCurve () const { return m_discount; }
    const ProjectCurveT& getProjectCurve (int spread_id) const { return m_proj; }
    ProjectCurveT& getBumpProjectCurve () { return m_proj; }
    DiscountCurve& getBumpDiscountCurve () { return m_discount; }

private:
    const LinearInterpDiscountCurve<mdouble> m_discount, m_proj;  
};
