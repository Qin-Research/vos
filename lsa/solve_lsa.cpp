#include <opengm/opengm.hxx>
#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/operations/minimizer.hxx>
#include <opengm/operations/adder.hxx>
#include <opengm/functions/potts.hxx>
#include <opengm/functions/pottsg.hxx>
#include <opengm/functions/truncated_squared_difference.hxx>
#include <opengm/functions/truncated_absolute_difference.hxx>
#include <opengm/graphicalmodel/space/simplediscretespace.hxx>
#include <opengm/graphicalmodel/graphicalmodel_hdf5.hxx>
#include <opengm/inference/lsatr.hxx>
#include <opengm/inference/auxiliary/minstcutkolmogorov.hxx>
#include <fstream>
#include <opengm/inference/graphcut.hxx>

using namespace std;
using namespace opengm;

int main(int argc, char *argv[])
{
    typedef opengm::DiscreteSpace<> Space;
    typedef opengm::ExplicitFunction<double> Function;
    typedef opengm::GraphicalModel<double,
            opengm::Adder,
            OPENGM_TYPELIST_5(opengm::ExplicitFunction<double> ,
                              opengm::PottsFunction<double>,
                              opengm::PottsGFunction<double>,
                              opengm::TruncatedSquaredDifferenceFunction<double>,
                              opengm::TruncatedAbsoluteDifferenceFunction<double>) , Space> GraphicalModelType;
    typedef opengm::LSA_TR<GraphicalModelType, opengm::Minimizer> LSATR;
typedef opengm::external::MinSTCutKolmogorov<size_t, double> MinStCutType;
typedef opengm::GraphCut<GraphicalModelType, opengm::Minimizer, MinStCutType> MinCut;

    GraphicalModelType gm;

    hdf5::load(gm, argv[1], argv[2]);


  //  MinCut lsa(gm);
    LSATR lsa(gm);
    lsa.infer();

    std::vector<GraphicalModelType::LabelType> labeling(gm.numberOfVariables());
    lsa.arg(labeling);

    //std::cout << gm.evaluate(labeling) << std::endl;

    std::ofstream ofs("label");
    for(int i = 0; i < labeling.size(); ++i)
    {
        ofs << labeling[i] << std::endl;
    }

    return 0;
}
