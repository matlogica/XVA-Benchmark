#include <atomic>
#include <string>
#include <iomanip> 

#include "XVAProblem.h"
#include "XVAJobRequest.h"
#include "DataTools.h"

// include adept sources in the same .cpp with main()
#include "adept/adept_source.h"

using json = nlohmann::json;

////////////////////////////////////////////////////
//
//  checkBondsFormula
//
//  This function allows to compare analytical Hull-White formula via MC simulation   
//
//  XVA      XVA problem
//
////////////////////////////////////////////////////

void checkBondsFormula(XVAProblem<double>& XVA) {
    std::mt19937_64 gen(17);
    std::normal_distribution<> normal_distrib(0, 1);

//    HullWhiteMarketModel<double> M=*XVA.getModel();
    std::vector<double> mean_rev_vector(XVA.getModel()->getMeanRev()->getVals());
    for (int i=0; i< mean_rev_vector.size(); i++) {
        mean_rev_vector[i]=(normal_distrib(gen)+10)/20;
    }
    HullWhiteMarketModel<double> model(
        XVA.getModel()->getR0(), XVA.getModel()->getSigma(), XVA.getModel()->getAlpha(),
        std::make_shared<PiecewiseLinearCurve<double>>(
            XVA.getModel()->getMeanRev()->getTimes(),
            mean_rev_vector
        ),
        XVA.getModel()->getSpreads()
    );
    model.initT0(XVA.getT0());
    double p(model.getIrMarket().getDiscountCurve()(20.)); // Analytical formula 
    
    std::cout << "Correctness of the HW-bond price formula \n";
    std::cout <<"HW-formula for 20 years bond price " << p << "\n";
    
    int mc_iterations=800;
    qtime time=20*365;
    int per_day=5;

    std::vector<double> normals(per_day*time);
    double bond=0;
    for (int i=0; i<mc_iterations; i++) {
        model.initT0(0);
        for (int j=0; j<normals.size(); j++) {
            normals[j]=normal_distrib(gen);
        }
        bond+=Vasicek<double>(time, model, normals);
    }
    std::cout << "MonteCarlo price " << bond/mc_iterations << "\n";
    std::cout << "Iterations=" << mc_iterations << "; partions_per_day=" << per_day << ";\n";
    std::cout << "relative formula/MC " << bond/mc_iterations/p << "\n";
    std::cout << "----------------------------\n";
}

////////////////////////////////////////////////////
//
//  checkHullWhiteFormula
//
//  This function allows to compare analytical Hull-White formula via MC simulation  
//
//  request_data     XVA task configuration
//
////////////////////////////////////////////////////

void checkHullWhiteFormula(const json& request_data) {
    std::cout<<"\n";
    XVAProblem<double> XVA_check; 
    XVA_check.initData(request_data);
    checkBondsFormula(XVA_check);
}

////////////////////////////////////////////////////
//
//  run_pricing
//
//  This function demonstrates how interface of XVAJobRequest can be used. 
//  There is a sequence of calls of XVA computations (processRequest()) for various 
//  portfolios and process parameters  
//
//  threads_num     number of threads 
//  path            path to the XVA task data
//
////////////////////////////////////////////////////

template<class mmType>
int run_pricing(const int threads_num, const std::string input_file) {
    json data_in, data_out;
    std::ifstream i(input_file);
    if(i.fail()) {
        std::cout << "Fail to open " << input_file << "\n";
        return 1;
    }
    i >> data_in;

    //checkHullWhiteFormula(data_in); 

    std::shared_ptr<std::vector<RequestFunction<mmType>>> func_request_cache = 
        std::make_shared<std::vector<RequestFunction<mmType>>>()
    ;
    std::shared_ptr<XVAJobRequest<mmType>> obj;
    std::atomic<bool> cancel= false;

    
    // Example of the XVA Task pricing loop
    obj=std::make_shared<XVAJobRequest<mmType>>(func_request_cache);
    obj->processRequest(data_in, data_out, threads_num, cancel);
    std::ofstream all_res("all_results.json");
    all_res << std::setw(4) << data_out << std::endl;
    all_res.close();
    
    return 0;
}

////////////////////////////////////////////////////
//
//  Main
//  
//  argv:
//  string    Path to XVA task data
//  int       AVX: 256/512
//  int       Number of threads
//
////////////////////////////////////////////////////

int main (int argc, char* argv[]) {
	int num_threads(1);
	std::string input_file("../Pricing/initData.json");
	if (argc > 3) num_threads = atoi(argv[3]);
    if (argc > 1) input_file=argv[1];
#if AADC_512 
    if (argc > 2 && atoi(argv[2]) == 512) return run_pricing<__m512d>(num_threads, input_file);
    else
#endif
    return run_pricing<__m256d>(num_threads, input_file);
}