#include "atomic"
#include "XVAProblem.h"
#include "XVAJobRequest.h"
#include "DataTools.h"
#include "string.h"
#include <iomanip> 
#include <thread>
#include "aadc/visitor_tools.h"

using json = nlohmann::json;

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

void collectVariableInputs (
    json& data, ArgumentsMap& request_variable_inputs
) {
    auto processOneInput = [&] (std::string str) {
        const auto json_path = json::json_pointer{str};
        idouble var=data[json_path].get<double>();
        request_variable_inputs[str]=std::make_pair(var, aadc::AADCArgument()); 
        
        data[json_path]=str;
    };

    processOneInput("/Currencies/EUR/ProjectionSpread.3M/flat_rate");
    processOneInput("/Currencies/EUR/ProjectionSpread.6M/flat_rate");
    processOneInput("/Currencies/EUR/ProjectionSpread.12M/flat_rate");
    processOneInput("/Currencies/EUR/HWMeanReversionCurve/flat_rate");
    processOneInput("/Currencies/EUR/r0");
    processOneInput("/Currencies/EUR/sigma");

    processOneInput("/CompanySurvivalCurve/flat_rate");
    processOneInput("/CounterPartySurvivalCurve/flat_rate");
}

////////////////////////////////////////////////////
//
//  markVariableInputs
//
//  request_variable_inputs     Variables dictionary    
//  diff                        true if derivative by this variable is required
//
////////////////////////////////////////////////////

void markVariableInputs(ArgumentsMap& request_variable_inputs, bool diff) {
    for (auto it = request_variable_inputs.begin(); it != request_variable_inputs.end(); ++it) {
        std::pair<idouble, aadc::AADCArgument>& idouble_arg_pair=it->second;
        if (diff) {
            idouble_arg_pair.second=idouble_arg_pair.first.markAsInput();
        } else {
            idouble_arg_pair.second=idouble_arg_pair.first.markAsInputNoDiff();
        }
    }
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
) {
    XVA.prepareSimulations();
    for (int mc_i=0; mc_i < mc_iterations; mc_i++) {
        XVA.simulatePath(randoms[mc_i]);
    }
    XVA.computeXVAMeasures();
    
    risk_results["CVA"][risk_id][risk_index] = (XVA.getCVA()/mc_iterations - base_results["CVA"].get<double>())/bump_size;
    risk_results["DVA"][risk_id][risk_index] = (XVA.getDVA()/mc_iterations - base_results["DVA"].get<double>())/bump_size;
}
