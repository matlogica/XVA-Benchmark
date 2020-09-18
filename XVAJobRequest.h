#pragma once
#include <atomic>
#include <string>
#include <iomanip> 
#include <thread>

#include "DataTools.h"
#include "XVAProblem.h"

#include "aadc/aadc.h"
#include "aadc/visitor_tools.h"

////////////////////////////////////////////////////
//
//  HWDiffResults
//
//  Holds gradient data of the HW model
//
//  <numberType>:     double, mmType, aadc:Argument
//
////////////////////////////////////////////////////

template<class numberType> 
class HWDiffResults { 
public:
    template<class Visitor, class T2>
    void visit(const Visitor& v, const HWDiffResults<T2>& other) {
        v.visit(sigma, other.sigma);
        v.visit(r0, other.r0);
        v.visit(mr_crv, other.mr_crv);
    }
public:
    typedef typename aadc::VectorType<numberType>::VecType VecType; 
    VecType mr_crv;
    numberType sigma, r0;
};

////////////////////////////////////////////////////
//
//  SurvDiffResults
//
//  Holds gradient data for survival curve
//
// <numberType>:     double, mmType, AADCArgument
//
////////////////////////////////////////////////////

template<class numberType>
class SurvDiffResults {
public:
    template<class Visitor, class T2>
    void visit(const Visitor& v, const SurvDiffResults<T2>& other) {
        v.visit(default_rates, other.default_rates);
    }
public:
    typedef typename aadc::VectorType<numberType>::VecType VecType; 
    VecType default_rates;
};

////////////////////////////////////////////////////
//
//  XVAResults
//
//  Stores data for the XVA results. 
//
//  <numberType>:  double, mmType, AADCResult
//
////////////////////////////////////////////////////

template<typename numberType>
class XVAResults {
public: 
    template<class Visitor, class T2>
    void visit(const Visitor& v, const XVAResults<T2>& other) {
        v.visit(CT, other.CT);
        v.visit(CVA, other.CVA);
        v.visit(DVA, other.DVA);
        v.visit(PEE, other.PEE);
        v.visit(NEE, other.NEE);
    }
public:
    typedef typename aadc::VectorType<numberType>::VecType VecType; 
    numberType CVA, DVA, CT;
    VecType PEE;
    VecType NEE;
};

////////////////////////////////////////////////////
//
//  XVADiffResults
//
//  Stores data for the XVA sensitivities. 
//
//  <numberType>:    double, mmType, AADCArgument
//
////////////////////////////////////////////////////

template<typename numberType> 
class XVADiffResults {
public:
    template<class Visitor, class T2>
    void visit(const Visitor& v, const XVADiffResults<T2>& other) {
        ir_crvs.resize(other.ir_crvs.size());
        for (int i=0; i < ir_crvs.size(); i++)  {
            ir_crvs[i].visit(v, other.ir_crvs[i]);
        }
        company_surv_crv.visit(v, other.company_surv_crv);
        cpty_surv_crv.visit(v, other.cpty_surv_crv);
    }
public: 
    std::vector<HWDiffResults<numberType> > ir_crvs;
    SurvDiffResults<numberType> company_surv_crv, cpty_surv_crv;
};

////////////////////////////////////////////////////
//
//  RequestFunction
// 
//  Structure holds cached AADC functions. This allows to reuse recorded functions
//  with a new marked data.
//
////////////////////////////////////////////////////

template<typename mmType>
struct RequestFunction {
    json modified_request_data;
    std::shared_ptr<aadc::AADCFunctions<mmType>> aad_funcs;
    ArgumentsMap request_variable_inputs;
    XVADiffResults<aadc::AADCArgument> xva_diff_args;
    XVAResults<aadc::AADCResult> res_args;
    aadc::VectorArg random_arg;
};

////////////////////////////////////////////////////
//
//  collectVariableInputs
//
//  This method allows to indicate active inputs of the future AADC-compiled function.
//  I.e. parameters which one can change and re-execute the AADC-function without its recompiling.
//  The whole logic allows user to use both XVAProblem<double> and XVAProblem<idouble> 
//  ProcessOneInput(str) transforms json data in a such a way that data[str]=str. 
//  If, after that procedure, data1==data2, then XVA computations can use the same AADC-Function.
//
//  data                         XVA task data 
//  request_variable_inputs      variables dictionary
//  
////////////////////////////////////////////////////
void collectVariableInputs (json& data, ArgumentsMap& request_variable_inputs);

////////////////////////////////////////////////////
//
//  markVariableInputs
//
//  request_variable_inputs     Variables dictionary    
//  diff                        true if derivative by this variable is required
//
////////////////////////////////////////////////////

void markVariableInputs (ArgumentsMap& request_variable_inputs, bool diff);

////////////////////////////////////////////////////
//
//  XVAJobRequest
//  
//  The logic of this class support XVA calculations. Function run_pricing demonstrates how it can be used.
//  in particular, method processRequest(json data) calls XVA computations for a given portfolio.
//  It stores compiled AADC functions and reuses it when possible
//
//  _func_request_cache    vector of RequestFunction<mmType>
//  _modified_jsons data   IR Model definition 
// 
////////////////////////////////////////////////////

template<class mmType>
class XVAJobRequest {
public:
    XVAJobRequest(
        std::shared_ptr<std::vector<RequestFunction<mmType>>>& _func_request_cache 
    ) 
        : m_func_request_cache(_func_request_cache)
    {}
    ~XVAJobRequest() {} 
    
    ////////////////////////////////////////////////////
    //
    //  init(const json& _data, int threads_num)
    //
    //  json _data:
    //  "MCPaths"                         int    Number of MC paths
    //  "UseMCPathBoundForPrimal"         bool   bound for number of MC paths used for primal and bump&revalue
    //  "MCPathsBoundForPrimal"           int    to be used if the previous is true
    //  "Forward Only"                    bool   compiles a pure replica of user valuations  
    //  "Primal Is Requred"               bool   if <double> version is required
    //  "Primal and Bumps Are Required"   bool   if sensitivities using bump&revalue are required 
    //  "Selective bumps"                 bool   if true, just 6 sensitivities will be computed only
    //  "Adept"                           bool   if sensitivities using the ADEPT are required 
    //  threads_num                              number of threads
    //
    ////////////////////////////////////////////////////

    void init(const json& _data, int threads_num) {
        m_data=_data;
        collectVariableInputs(m_data, m_request_variable_inputs); // getArgumentsMap());   
     
        m_mc_iterations=m_data["MCPaths"].template get<int>();
        // X -> nearest upper multiple of AVX_size*threads_num
        const int threads=m_AVX_size*threads_num;
        m_mc_iterations += (threads-1) - (m_mc_iterations+(threads-1)) % threads; 
        m_AVX_iterations=m_mc_iterations/(m_AVX_size*threads_num);

        const bool use_mc_path_bound = m_data["UseMCPathBoundForPrimal"].template get<bool>();
        const int MC_path_bound=m_data["MCPathsBoundForPrimal"].template get<int>();
        m_primal_mc_iterations = m_mc_iterations;
        if (use_mc_path_bound && (MC_path_bound < m_mc_iterations)) m_primal_mc_iterations=MC_path_bound;
        m_norm_coeff=double(m_mc_iterations)/double(m_primal_mc_iterations);

        m_all_results=json::object();
 
        m_forward_only=m_data["Forward Only"].template get<bool>();
        m_primal_is_required=m_data["Primal Is Requred"].template get<bool>();
        m_adept_is_required=m_data["Adept"].template get<bool>();
        m_primal_and_bumps_are_required=m_data["Primal and Bumps Are Required"].template get<bool>();
        m_selective_bumps=m_data["Selective bumps"].template get<int>();
    }

    //////////////////////////////////////////////////////
    //
    //  generateRandoms(int num_randoms_per_path)
    //
    //  generate num_randoms_per_path*Monte_Carlo_paths numbers
    //  
    //  num_randoms_per_path     Number of random numbers per one MC path
    //
    //////////////////////////////////////////////////////

    void generateRandoms(int num_randoms_per_path) {
        m_randoms=std::vector<std::vector<double>>();
        m_mm_randoms=std::vector<mmVector<mmType>>();
        std::mt19937_64 gen(17);
        std::normal_distribution<> normal_distrib(0, 1);
        for (int mc_i=0; mc_i<m_mc_iterations; mc_i++) {
            std::vector<double> random_vec;
            for(int i=0; i < num_randoms_per_path; i++) {
                random_vec.push_back(normal_distrib(gen));
            }
            m_randoms.push_back(random_vec);
        }
        aadc::restructurizeData(m_randoms, m_mm_randoms);
    }

    // Method  processRequest() manages computation of xVAs and its sensitivities for the new_data.
    void processRequest(const json& request_data, json& data_out, int threads_num, std::atomic<bool>& cancel);
    // primal() implements a <double> replica of xVA computations
    void primal(const json& request_data);
    // compileAADFunction() implements compilation of AADC-functions for xVAs computations
    void compileAADFunction(const json& request_data);
    // aADExecution() implements computation of the  vectorized AAD-function in a MT regime
    void aADExecution(const json& request_data, int threads_num, std::atomic<bool>& cancel);    
    // bumpAndRevalue() implements computation of xVAs sensitivities using bump&revalue method
    void bumpAndRevalue(const json& request_data);
    //adept() implements computations of xVAs sensitivites using ADEPT AAD library
    void adept(const json& request_data);

public:
    json m_data;
    int m_mc_iterations, m_AVX_size, m_AVX_iterations, m_primal_mc_iterations; 
    double m_norm_coeff;
    bool m_primal_is_required, m_primal_and_bumps_are_required, m_selective_bumps
        , m_forward_only, m_adept_is_required
    ;
    std::vector<std::vector<double>> m_randoms;
    std::vector<mmVector<mmType>> m_mm_randoms;
 
    std::shared_ptr<std::vector<RequestFunction<mmType>>> m_func_request_cache;

    std::shared_ptr<aadc::AADCFunctions<mmType>> m_aad_funcs;
    ArgumentsMap m_request_variable_inputs;
    XVADiffResults<aadc::AADCArgument> m_xva_diff_args;
    XVAResults<aadc::AADCResult> m_res_args;
    aadc::VectorArg m_random_arg;

    json m_aad_results, m_all_results;
    json m_bump_risk_results, m_aad_risk_results; 
    std::chrono::microseconds m_base_time, m_adept_base_time, m_aad_time;
};

////////////////////////////////////////////////////
//
//  XVAJobRequest<mmType>::primal
//
//  Calculates XVA measures using native <double> type.
//
//  request_data     XVA task configuration
//
////////////////////////////////////////////////////

template<class mmType>
void XVAJobRequest<mmType>::primal(const json& request_data) {
    double c_t_accumulated=0;

    XVAProblem<double> XVA;
    XVA.initData(request_data);   
    generateRandoms(XVA.numberOfRandomVars());
    auto base_start = std::chrono::high_resolution_clock::now();
        XVA.prepareSimulations();
        for (int mc_i=0; mc_i < m_primal_mc_iterations; mc_i++) {
            XVA.simulatePath(m_randoms[mc_i]);
            c_t_accumulated+=XVA.getCSA()->getCollateral();
        }
        XVA.computeXVAMeasures();
        c_t_accumulated/=m_primal_mc_iterations;
        std::vector<double> pee_res(XVA.getPEE()), nee_res(XVA.getNEE());
        for (int i=0; i < pee_res.size(); i++) {
            pee_res[i]/=m_primal_mc_iterations;
            nee_res[i]/=m_primal_mc_iterations;
        }
    auto base_stop= std::chrono::high_resolution_clock::now();
    
    XVA.m_xva_data["DVA"]=XVA.getDVA()/m_primal_mc_iterations;
    XVA.m_xva_data["CVA"]=XVA.getCVA()/m_primal_mc_iterations;
    XVA.m_xva_data["PEE"]=pee_res;
    XVA.m_xva_data["NEE"]=nee_res;
    
    std::cout << "Total number of iterations: " << m_mc_iterations << "\n";
    std::cout << "CVA " << XVA.m_xva_data["CVA"] << "\n";
    std::cout << "DVA " << XVA.m_xva_data["DVA"] << "\n";
    std::cout << "Collateral.back " << c_t_accumulated<< "\n";
    
    m_base_time=
        std::chrono::duration_cast<std::chrono::microseconds>(base_stop - base_start)
    ;
    std::cout << "Base time (double): " << m_mc_iterations << " iterations(actual num simulations " 
        << m_primal_mc_iterations << ") = " 
        << m_base_time.count()*m_norm_coeff << " microseconds\n"
    ;    
    // end of double time measurement
    m_all_results["primal time"]=m_base_time.count()*m_norm_coeff;

    std::ofstream dbl_res_aad("res_dbl.json");
    dbl_res_aad << std::setw(4) << XVA.m_xva_data << std::endl;

    m_all_results["primal"]= XVA.m_xva_data;
}

////////////////////////////////////////////////////
//
//  XVAJobRequest<mmType>::compileAADFunction
//
//  Compilation of AADC-functions for xVAs computations
//
//  request_data     XVA task configuration
//
////////////////////////////////////////////////////

template <class mmType>
void XVAJobRequest<mmType>::compileAADFunction(const json& request_data) {
    auto cmpl_start = std::chrono::high_resolution_clock::now();

    XVAProblem<double> XVA;
    XVA.initData(request_data);     
    // vector of random variables used for simulation of one path
    std::vector<idouble> aad_random_vec(XVA.numberOfRandomVars());

    getArgumentsMap() = m_request_variable_inputs;
    
    m_aad_funcs->startRecording();
        // Mark vector of random variables as input only. No adjoints for them
        markVectorAsInput(m_random_arg, aad_random_vec, false);          
        
        // Mark all JSON/input parameters
        markVariableInputs(getArgumentsMap(), !m_forward_only);
        
        // If one wishes to calculate derivatives relative to any objects, thus these objects should be created here:
        // i.e. between startRecording() and a Checkpoint.
        XVAProblem<idouble> aad_XVA; 
        aad_XVA.initData(m_data);    
            
        idouble::CheckPoint();
        // Now we should mark these created variables. 
        if (!m_forward_only) {
            m_xva_diff_args  =XVADiffResults<aadc::AADCArgument>();
            m_xva_diff_args.ir_crvs.push_back(HWDiffResults<aadc::AADCArgument>());

            m_xva_diff_args.ir_crvs[0].r0= aad_XVA.getModel()->getR0().markAsDiff();
            m_xva_diff_args.ir_crvs[0].sigma= aad_XVA.getModel()->getSigma().markAsDiff();

            for (int i=0; i<aad_XVA.getModel()->getMeanRev()->getVals().size(); i++) {
                m_xva_diff_args.ir_crvs[0].mr_crv.
                    push_back(aad_XVA.getModel()->getMeanRev()->getVals()[i].markAsDiff())
                ;
            }
            for (int i=0; i<aad_XVA.getCompSurvCurv()->getZeroRatesVector().size(); i++) {
                m_xva_diff_args.company_surv_crv.default_rates.
                    push_back(aad_XVA.getCompSurvCurv()->getZeroRatesVector()[i].markAsDiff())
                ;
            }
            for (int i=0; i<aad_XVA.getCtrpSurvCurv()->getZeroRatesVector().size(); i++) {
                m_xva_diff_args.cpty_surv_crv.default_rates.
                    push_back(aad_XVA.getCtrpSurvCurv()->getZeroRatesVector()[i].markAsDiff())
                ;
            }
        }
        // Record one path
        aad_XVA.prepareSimulations();
        aad_XVA.simulatePath(aad_random_vec);
        aad_XVA.computeXVAMeasures();
        
        // Mark computed measures as output
        m_res_args.CVA = aad_XVA.getCVA().markAsOutput();
        m_res_args.DVA = aad_XVA.getDVA().markAsOutput();
        m_res_args.CT = aad_XVA.getCSA()->getCollateral().markAsOutput();
        markVectorAsOutput(m_res_args.PEE, aad_XVA.getPEE());
        markVectorAsOutput(m_res_args.NEE, aad_XVA.getNEE());
    m_aad_funcs->stopRecording();
    
    auto cmpl_stop = std::chrono::high_resolution_clock::now();
    
    std::chrono::microseconds cmpl_time=
        std::chrono::duration_cast<std::chrono::microseconds>(cmpl_stop - cmpl_start)
    ;
    std::cout << "\n------AAD-Compiler-------------\n";
    std::cout << "Compilation time = " << cmpl_time.count()<< " microseconds\n";    
    std::cout << "Compilation time / (Base time / Iterations) = " << cmpl_time.count() / 
        (m_base_time.count()/m_primal_mc_iterations)<< " Base iteration times\n";   
    
    std::cout << "Code size forward : " << m_aad_funcs->getCodeSizeFwd() << std::endl;
    std::cout << "Code size reverse : " << m_aad_funcs->getCodeSizeRev() << std::endl;
    std::cout << "Work array size   : " << m_aad_funcs->getWorkArraySize() << std::endl;
    std::cout << "Stack size        : " << m_aad_funcs->getStackSize() << std::endl;
    std::cout << "Const data size   : " << m_aad_funcs->getConstDataSize() << std::endl;
    std::cout << "CheckPoint size   : " << m_aad_funcs->getNumCheckPointVars() << std::endl;

    m_all_results["compiler data"]["Compilation time"]= cmpl_time.count();    
    m_all_results["compiler data"]["Compilation time / (Base time / Iterations)"]=cmpl_time.count() / 
        (m_base_time.count()/m_primal_mc_iterations);   
    
    m_all_results["compiler data"]["Code size forward"]= m_aad_funcs->getCodeSizeFwd();
    m_all_results["compiler data"]["Code size reverse"]= m_aad_funcs->getCodeSizeRev();
    m_all_results["compiler data"]["Work array size"]= m_aad_funcs->getWorkArraySize();
    m_all_results["compiler data"]["Stack size"]=  m_aad_funcs->getStackSize();
    m_all_results["compiler data"]["Const data size"]=  m_aad_funcs->getConstDataSize();
    m_all_results["compiler data"]["CheckPoint size "]= m_aad_funcs->getNumCheckPointVars();
}

//////////////////////////////////////////////
//
//  XVAJobRequest<mmType>::aADExecution
//
//  Computes XVA measures and risks to model parameters using AADC Functions
//
//  request_data     XVA task configuration
//  threads_num      Number of threads
//  cancel           Flag to cancel MonteCarlo simulation
//
///////////////////////////////////////////

template <class mmType>
void XVAJobRequest<mmType>::aADExecution(const json& request_data, int threads_num, std::atomic<bool>& cancel) {
    std::cout << "Num Threads: " << threads_num <<"\n";
    std::cout << "AVX level: avx" << sizeof(mmType)*8 << std::endl;

    XVAResults<double> d_results;
    XVADiffResults<double> CVA_diff, DVA_diff; 

    d_results.visit(aadc::InitializeVisitor<double>(), m_res_args);
    CVA_diff.visit(aadc::InitializeVisitor<double>(), m_xva_diff_args);
    DVA_diff.visit(aadc::InitializeVisitor<double>(), m_xva_diff_args);

    std::vector<XVAResults<double>> thread_results(threads_num,d_results);
    std::vector<XVADiffResults<double>> thread_CVA_diff(threads_num, CVA_diff);
    std::vector<XVADiffResults<double>> thread_DVA_diff(threads_num, DVA_diff);

    auto threadWorker = [&] (
        XVAResults<double>& d_results,
        XVADiffResults<double>& CVA_diff,
        XVADiffResults<double>& DVA_diff, 
        const ArgumentsMap& request_variable_inputs,
        const int th_i 
    ) {
        std::shared_ptr<aadc::AADCWorkSpace<mmType> > ws(m_aad_funcs->createWorkSpace());       
        
        XVAResults<mmType> mmXVA_res;
        XVADiffResults<mmType> mmCVA_diff, mmDVA_diff;
        
        d_results.visit(aadc::InitializeVisitor<double>(), m_res_args);
        CVA_diff.visit(aadc::InitializeVisitor<double>(), m_xva_diff_args);
        DVA_diff.visit(aadc::InitializeVisitor<double>(), m_xva_diff_args);

        mmXVA_res.visit(aadc::InitializeVisitor<mmType>(), m_res_args);
        mmCVA_diff.visit(aadc::InitializeVisitor<mmType>(), m_xva_diff_args);
        mmDVA_diff.visit(aadc::InitializeVisitor<mmType>(), m_xva_diff_args);

        for (auto i=request_variable_inputs.begin(); i!=request_variable_inputs.end(); ++i) {
            const auto json_path = json::json_pointer{i->first};
            ws->val(i->second.second)=aadc::mmSetConst<mmType>(request_data[json_path]);
        }
        m_aad_funcs->forward(*ws,0,0);
        //MC starts

        for (int mc_i=0; (mc_i < m_AVX_iterations) && !cancel; mc_i++) {
            // random numbers for this path
            setAVXVector(*ws, m_random_arg, m_mm_randoms[mc_i+m_AVX_iterations*th_i]);
            m_aad_funcs->forward(*ws,1,-1);

            // collect output results
            aadc::MMAccumulateVisitorWS<mmType> accum_visitor(*ws);
            mmXVA_res.visit(accum_visitor, m_res_args);

            if (!m_forward_only) {
                // perform multiple reverse valuations.
                // One for each output value
                //CVA
                ws->resetDiff();
                ws->diff(m_res_args.CVA)=aadc::mmSetConst<mmType>(1);
                ws->diff(m_res_args.DVA)=aadc::mmSetConst<mmType>(0);
                m_aad_funcs->reverse(*ws, 1,-1);
                mmCVA_diff.visit(accum_visitor, m_xva_diff_args);

                //DVA
                ws->resetDiff();
                ws->diff(m_res_args.CVA)=aadc::mmSetConst<mmType>(0);
                ws->diff(m_res_args.DVA)=aadc::mmSetConst<mmType>(1);
                m_aad_funcs->reverse(*ws,1,-1);
                mmDVA_diff.visit(accum_visitor, m_xva_diff_args);
            }            
        }
        
        // sum over AVX components of the avx results and its sensitivities
        d_results.visit(aadc::MMSumReduceVisitor<mmType>(1./m_mc_iterations), mmXVA_res);

        if (!m_forward_only) {
            CVA_diff.visit(aadc::MMSumReduceVisitor<mmType>(1./m_mc_iterations), mmCVA_diff);
            DVA_diff.visit(aadc::MMSumReduceVisitor<mmType>(1./m_mc_iterations), mmDVA_diff);
        }
    };
        
    auto aad_start = std::chrono::high_resolution_clock::now();
        std::vector<std::unique_ptr<std::thread>> threads;
        for(int i=0; i< threads_num; i++) {
            threads.push_back(
                std::unique_ptr<std::thread>(
                    new std::thread(
                        threadWorker
                        , std::ref(thread_results[i])
                        , std::ref(thread_CVA_diff[i])
                        , std::ref(thread_DVA_diff[i])
                        , std::ref(m_request_variable_inputs)
                        , i
                    )
                )
            );
        }
        for(auto&& t: threads) t->join();
        
        for (int i=1; i< threads_num; i++) {
            thread_results[0].visit(aadc::AccumulateVisitor<double>(), thread_results[i]);
            thread_CVA_diff[0].visit(aadc::AccumulateVisitor<double>(), thread_CVA_diff[i]);
            thread_DVA_diff[0].visit(aadc::AccumulateVisitor<double>(), thread_DVA_diff[i]);
    }
    auto aad_stop= std::chrono::high_resolution_clock::now();
    // end of aad time measurement    
    m_aad_time=
        std::chrono::duration_cast<std::chrono::microseconds>(aad_stop - aad_start)
    ;
    
    std::cout << "Base time normalization coefficient: " << m_norm_coeff << "\n";
    std::cout << "AADC : " << m_mc_iterations << " iterations = " << m_aad_time.count()<< " microseconds\n";
    
    if (m_primal_is_required) {
        std::string task = !m_forward_only ?  "(Fw+Rev(CVA)+Rev(DVA))" : "Fw";
        std::cout << "Relative performance " << task << "/Primal is " << 
            double(m_aad_time.count()) / (double(m_base_time.count()) * m_norm_coeff)<< " times\n"
        ;    
        m_all_results["Relative performance "]=double(m_aad_time.count()) / 
            (double(m_base_time.count())* m_norm_coeff)
        ;
    }

    m_aad_results["CVA"]=thread_results[0].CVA;
    m_aad_results["DVA"]=thread_results[0].DVA;
    
    std::cout << std::setprecision(14);
    std::cout << "AADC CVA: "  << thread_results[0].CVA << 
        ". Compare with primal result: " << m_all_results["primal"]["CVA"].template get<double>() << "\n";
    std::cout << "aadc-CVA - primal-CVA: " << 
        thread_results[0].CVA - m_all_results["primal"]["CVA"].template get<double>() << "\n";
    std::cout << "AADC DVA: "  << thread_results[0].DVA << 
        ". Compare with primal result: " << m_all_results["primal"]["DVA"].template get<double>() << "\n";
    std::cout << "aadc-DVA - primal-DVA: " << 
        thread_results[0].DVA - m_all_results["primal"]["DVA"].template get<double>() << "\n";

    m_aad_results["PEE"]=thread_results[0].PEE;
    m_aad_results["NEE"]=thread_results[0].NEE;
    
    std::ofstream res_aad("res_aad.json");
    res_aad << std::setw(4) << m_aad_results << std::endl;

    m_all_results["AADC results"]=m_aad_results;
    
    if (!m_forward_only) {
        for (int i=0; i < m_xva_diff_args.ir_crvs[0].mr_crv.size(); i++) {
            m_aad_risk_results["CVA"]["MeanRev"][i]=thread_CVA_diff[0].ir_crvs[0].mr_crv[i];
            m_aad_risk_results["DVA"]["MeanRev"][i]=thread_DVA_diff[0].ir_crvs[0].mr_crv[i];
        }

        for (int i=0; i < m_xva_diff_args.cpty_surv_crv.default_rates.size(); i++) {
            m_aad_risk_results["CVA"]["CtrpSurv"][i]=thread_CVA_diff[0].cpty_surv_crv.default_rates[i];
        }
        for (int i=0; i < m_xva_diff_args.company_surv_crv.default_rates.size(); i++) {
            m_aad_risk_results["DVA"]["CompSurv"][i]=thread_DVA_diff[0].company_surv_crv.default_rates[i];
        }
        m_aad_risk_results["CVA"]["sigma"][0]=thread_CVA_diff[0].ir_crvs[0].sigma;
        m_aad_risk_results["CVA"]["r0"][0]=thread_CVA_diff[0].ir_crvs[0].r0;
        m_aad_risk_results["DVA"]["sigma"][0]=thread_DVA_diff[0].ir_crvs[0].sigma;
        m_aad_risk_results["DVA"]["r0"][0]=thread_DVA_diff[0].ir_crvs[0].r0;
    }

    std::ofstream o_aad("aad_out.json");
    o_aad << std::setw(4) << m_aad_risk_results << std::endl;
    m_all_results["AAD_risks"]=m_aad_risk_results;
}

////////////////////////////////////////////////////
//
//  calcRiskByBump
//
//  Delivers one sensitivity using bump&revalue methods
//
//  risk_results     XVA task configuration
//  base_results     Results of the primal algorithm  
//  XVA              XVA problem with one shocked parameter 
//  risk_id          Principal part of the address in the json
//  risk_index       Auxiliary part of the address in the json
//  bump_size        Bump size
//  mc_iterations    Number of Monte Carlo iterations
//  randoms          2D vector of random numbers
//
////////////////////////////////////////////////////

void calcRiskByBump (
    json& risk_results, 
    const json& base_results, 
    XVAProblem<double>& XVA, 
    const std::string& risk_id, 
    const int risk_index,
    const double bump_size,
    const int mc_iterations,
    const std::vector<std::vector<double>>& randoms
);

////////////////////////////////////////////////////
//
//  XVAJobRequest<mmType>::calcRiskByBump
//
//  Implements computation of sensitivities using bump&revalue methods
//
//  request_data     XVA task configuration
//
////////////////////////////////////////////////////

template<class mmType>
void XVAJobRequest<mmType>::bumpAndRevalue(const json& request_data) {
    double bump_size=0.00000001;
    if (!m_selective_bumps) {
        XVAProblem<double> xva;
        xva.initData(request_data);

        for (int i=0; i < xva.getModel()->getMeanRev()->getVals().size(); i++) {
            json bump_data(request_data);
            bump_data["Currencies"]["EUR"]["HWMeanReversionCurve"]["bump_index"]=i;
            bump_data["Currencies"]["EUR"]["HWMeanReversionCurve"]["bump_size"]=bump_size;
            XVAProblem<double> bumped_xva;
            bumped_xva.initData(bump_data);
            calcRiskByBump(
                m_bump_risk_results, m_all_results["primal"], bumped_xva
                , "MeanRev", i, bump_size, m_primal_mc_iterations, m_randoms
            );
        }
    
        for (int i=0; i < xva.getCtrpSurvCurv()->getZeroRatesVector().size(); i++) {
            json bump_data(request_data);
            bump_data["CounterPartySurvivalCurve"]["bump_index"]=i;
            bump_data["CounterPartySurvivalCurve"]["bump_size"]=bump_size;
            XVAProblem<double> bumped_xva;
            bumped_xva.initData(bump_data);
            calcRiskByBump(
                m_bump_risk_results, m_all_results["primal"], bumped_xva
                , "CtrpSurv", i, bump_size, m_primal_mc_iterations, m_randoms
            );
        }
        
        for (int i=0; i < xva.getCompSurvCurv()->getZeroRatesVector().size(); i++) {
            json bump_data(request_data);
            bump_data["CompanySurvivalCurve"]["bump_index"]=i;
            bump_data["CompanySurvivalCurve"]["bump_size"]=bump_size;
            XVAProblem<double> bumped_xva;
            bumped_xva.initData(bump_data);
            calcRiskByBump(
                m_bump_risk_results, m_all_results["primal"], bumped_xva
                , "CompSurv", i, bump_size, m_primal_mc_iterations, m_randoms
            );
        }
    } else {
        json bump_data(request_data);
        bump_data["Currencies"]["EUR"]["HWMeanReversionCurve"]["bump_index"]=3;
        bump_data["Currencies"]["EUR"]["HWMeanReversionCurve"]["bump_size"]=bump_size;
        XVAProblem<double> bumped_xva;
        bumped_xva.initData(bump_data);

        calcRiskByBump(
            m_bump_risk_results, m_all_results["primal"], bumped_xva
            , "MeanRev", 3, bump_size, m_primal_mc_iterations, m_randoms
        );
    }
    //Sigma 
    json bump_data(request_data);
    double base_value=bump_data["Currencies"]["EUR"]["sigma"].get<double>();
    bump_data["Currencies"]["EUR"]["sigma"]=base_value+bump_size;
    XVAProblem<double> bumped_xva;
    bumped_xva.initData(bump_data);

    calcRiskByBump(
        m_bump_risk_results, m_all_results["primal"], bumped_xva
        , "sigma", 0, bump_size, m_primal_mc_iterations, m_randoms
    );

    //r0
    bump_data=request_data;
    base_value=bump_data["Currencies"]["EUR"]["r0"].get<double>();
    bump_data["Currencies"]["EUR"]["r0"]=base_value+bump_size;
    bumped_xva=XVAProblem<double>();
    bumped_xva.initData(bump_data);

    calcRiskByBump(
        m_bump_risk_results, m_all_results["primal"], bumped_xva
        , "r0", 0, bump_size, m_primal_mc_iterations, m_randoms
    );

    std::ofstream o_bump("bump_out.json");
    o_bump << std::setw(4) << m_bump_risk_results << std::endl;
    m_all_results["bump&revalue risks"]=m_bump_risk_results;
}

////////////////////////////////////////////////////
//
//  XVAJobRequest<mmType>::processRequest
//
//  Method  processRequest() manages computation of xVAs and its sensitivities for the new_data. 
//  It calls method primal(), that is a <double> replica of xVA computations,
//  then it check function's cache if any of its AADC_functions can be reused, 
//  otherwise a new AADC-function would be compiled (compileAADFunction()) and stored in this cache. 
//  then it runs the vectorized AAD-function in a MT regime (aADExecution()). Finally the processRequest() method
//  computes sensitivities using bump&revalue (bumpAndRevalue()) and using ADEPT package (adept()) as well.
//
//  request_data     XVA task configuration
//  data_out         XVA task output
//  threads_num      number of threads
//  cancel           
//
////////////////////////////////////////////////////

template<class mmType>
void XVAJobRequest<mmType>::processRequest(
    const json& request_data
    , json& data_out
    , const int threads_num
    , std::atomic<bool>& cancel
) {
    std::cout << "----------------------------\n";
    m_AVX_size = sizeof(mmType) / sizeof(double);
    init(request_data, threads_num);
    if (m_primal_is_required) primal(request_data);
    // Look if required structure exists in the cache already. Otherwise a new AADC-function will be compiled. 
    auto cached_func_i = m_func_request_cache->begin();
    while (cached_func_i != m_func_request_cache->end() && cached_func_i->modified_request_data != m_data) ++cached_func_i;
    if (cached_func_i != m_func_request_cache->end()) {
        std::cout << " REUSE FUNCTION NUMBER " << cached_func_i-m_func_request_cache->begin()  << " \n";

        m_aad_funcs=cached_func_i->aad_funcs;            
        m_request_variable_inputs = cached_func_i->request_variable_inputs;
        m_res_args = cached_func_i->res_args;
        m_xva_diff_args = cached_func_i->xva_diff_args;
        m_random_arg = cached_func_i->random_arg;
    }
    else {
        std::cout << "CREATE NEW FUNCTION. Number "  << m_func_request_cache->size() << "\n";
        m_aad_funcs=std::shared_ptr<aadc::AADCFunctions<mmType>>(
            new aadc::AADCFunctions<mmType>(
                {
                    {aadc::AADC_NumCompressorThreads, 9}
                    , {aadc::AADC_InitInputDiff, 0}   // not reset input_diff
                }
            )
        );
        compileAADFunction(request_data);
        // Store data&objects for further reusing  
        RequestFunction<mmType> temp_obj; 
        temp_obj.modified_request_data = m_data;
        m_request_variable_inputs = temp_obj.request_variable_inputs=getArgumentsMap();
        temp_obj.aad_funcs=m_aad_funcs;
        temp_obj.res_args=m_res_args;
        temp_obj.xva_diff_args=m_xva_diff_args;
        temp_obj.random_arg=m_random_arg;
    
        m_func_request_cache->push_back(temp_obj);
    }
    aADExecution(request_data, threads_num, cancel);
    if (m_primal_is_required && m_adept_is_required) adept(request_data);
    if (m_primal_and_bumps_are_required) bumpAndRevalue(request_data);
    data_out=m_all_results;
}

////////////////////////////////////////////////////
//
//  XVAJobRequest<mmType>::adept
//
//  Implements computations of xVAs sensitivites using ADEPT AAD library 
//
//  request_data    XVA task configuration
//
////////////////////////////////////////////////////

template<class mmType>
void XVAJobRequest<mmType>::adept(const json& request_data) {
    adept::Stack stack_;
    double c_t_accumulated=0;
    auto adept_base_start = std::chrono::high_resolution_clock::now();
        double CVA_res=0, dva_res = 0.;
        double CVA_sigma(0.), DVA_sigma(0.);
        double CVA_r0(0.), DVA_r0(0.);
        std::vector<adept::adouble> adept_randoms(m_randoms[0].size());
        
        XVAProblem<adept::adouble> adept_XVA;
        adept_XVA.initData(request_data);            
        for (int mc_i=0; mc_i<m_primal_mc_iterations; mc_i++) {
            set_values(&adept_randoms[0], m_randoms[0].size(), &(m_randoms[mc_i][0]));
            stack_.new_recording(); 
            adept_XVA.prepareSimulations();
            
            adept_XVA.simulatePath(adept_randoms);
            c_t_accumulated += adept_XVA.getCSA()->getCollateral().value();
            adept_XVA.computeXVAMeasures();
            CVA_res += adept_XVA.getCVA().value();
            dva_res += adept_XVA.getDVA().value();
            
            adept_XVA.getCVA().set_gradient(1.); // only one J row here
            adept_XVA.getDVA().set_gradient(0.);
                
            stack_.reverse();
            CVA_sigma += adept_XVA.getModel()->getSigma().get_gradient();
            CVA_r0 += adept_XVA.getModel()->getR0().get_gradient();
            stack_.clear_gradients();
            adept_XVA.getCVA().set_gradient(0.);
            adept_XVA.getDVA().set_gradient(1.);
            
            stack_.reverse();
            DVA_sigma += adept_XVA.getModel()->getSigma().get_gradient();
            DVA_r0 += adept_XVA.getModel()->getR0().get_gradient();
        }
        c_t_accumulated/=m_primal_mc_iterations;
        std::vector<double> pee_res(adept_XVA.getPEE().size()), nee_res(adept_XVA.getNEE().size());

        for (int i=0; i<adept_XVA.getPEE().size(); i++) {
            pee_res[i]=adept_XVA.getPEE()[i].value() / m_primal_mc_iterations;
            nee_res[i]=adept_XVA.getNEE()[i].value() / m_primal_mc_iterations;
        }
    auto adept_base_stop= std::chrono::high_resolution_clock::now();
    
    adept_XVA.m_xva_data["DVA"]=dva_res/m_primal_mc_iterations;
    adept_XVA.m_xva_data["CVA"]=CVA_res/m_primal_mc_iterations;
    adept_XVA.m_xva_data["PEE"]=pee_res;
    adept_XVA.m_xva_data["NEE"]=nee_res;
    
    std::cout << "----------ADEPT---------\n";
    std::cout << "Total number of iterations: " << m_mc_iterations << "\n";
    std::cout << "CVA " << adept_XVA.m_xva_data["CVA"] << "\n";
    std::cout << "DVA " << adept_XVA.m_xva_data["DVA"] << "\n";
    std::cout << "adept: CVA by sigma " << CVA_sigma/m_primal_mc_iterations << "\n";
    std::cout << "adept: CVA by r0 " << CVA_r0/m_primal_mc_iterations << "\n";
    std::cout << "adept: DVA by sigma " << DVA_sigma/m_primal_mc_iterations << "\n";
    std::cout << "adept: DVA by r0 " << DVA_r0/m_primal_mc_iterations << "\n";
    std::cout << "--------compare Adept and AADC risk results---------\n";

    std::cout << "Adept::CVAsigma - AADC::CVAsigma: " << 
        CVA_sigma/m_primal_mc_iterations- m_aad_risk_results["CVA"]["sigma"][0].template get<double>() << "\n";
    std::cout << "Adept::DVAsigma - AADC::DVAsigma: " << 
        DVA_sigma/m_primal_mc_iterations- m_aad_risk_results["DVA"]["sigma"][0].template get<double>() << "\n";
    std::cout << "Adept::CVAr0 - AADC::CVAr0: " << 
        CVA_r0/m_primal_mc_iterations- m_aad_risk_results["CVA"]["r0"][0].template get<double>() << "\n";
    std::cout << "Adept::DVAr0 - AADC::DVAr0: " << 
        DVA_r0/m_primal_mc_iterations- m_aad_risk_results["DVA"]["r0"][0].template get<double>() << "\n";
    
    m_adept_base_time=
        std::chrono::duration_cast<std::chrono::microseconds>(adept_base_stop - adept_base_start)
    ;

    std::cout << "Adept/Base time: " << double(m_adept_base_time.count())/double(m_base_time.count()) << "\n";
    std::cout << "Adept/AADC time: " 
        << double(m_adept_base_time.count()*m_norm_coeff)/double(m_aad_time.count()) 
        << "\n"
    ;
    std::cout << "Adept time (double): " << m_mc_iterations << " iterations(actual num simulations " 
        << m_primal_mc_iterations << ") = " 
        << m_adept_base_time.count()*m_norm_coeff << " microseconds\n"
    ;    
    // end of Adept time measurement
}