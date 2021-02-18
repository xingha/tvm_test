#include <chrono>
#include <iostream>
#include <fstream>
#include <typeinfo>
#include "opencv2/opencv.hpp"
#include "tvm/runtime/module.h"
#include "tvm/runtime/registry.h"
#include "tvm/runtime/packed_func.h"

typedef std::chrono::high_resolution_clock Clock;

static std::string libPath = "tvm-model/deploy_cuda_lib.so";
static std::string graphPath = "tvm-model/deploy_cuda_graph.json";
static std::string paramPath = "tvm-model/deploy_cuda_param.params";

class FR_MFN_Deploy{
private:
    void * handle;

public:
    FR_MFN_Deploy(std::string modelFolder)
    {
        tvm::runtime::Module mod_syslib = tvm::runtime::Module::LoadFromFile(libPath);
        //load graph
        std::string modelPath = modelFolder + graphPath;
        std::ifstream json_in(modelPath);
        std::string json_data((std::istreambuf_iterator<char>(json_in)), std::istreambuf_iterator<char>());
        json_in.close();
        int device_type = kDLCPU;
        int device_id = 0;
        // get global function module for graph runtime
        tvm::runtime::Module mod = (*tvm::runtime::Registry::Get("tvm.graph_runtime.create"))(json_data, mod_syslib, device_type, device_id);
        this->handle = new tvm::runtime::Module(mod);
        //load param
        std::ifstream params_in(modelFolder + paramPath, std::ios::binary);
        std::string params_data((std::istreambuf_iterator<char>(params_in)), std::istreambuf_iterator<char>());
        params_in.close();
        TVMByteArray params_arr;
        params_arr.data = params_data.c_str();
        params_arr.size = params_data.length();
        tvm::runtime::PackedFunc load_params = mod.GetFunction("load_params");
        load_params(params_arr);
    }


    cv::Mat forward(cv::Mat inputImageAligned)
    {
        //mobilefacnet preprocess has been written in graph.
        cv::Mat tensor = cv::dnn::blobFromImage(inputImageAligned,1.0,cv::Size(112,112),cv::Scalar(0,0,0),true);
        //convert uint8 to float32 and convert to RGB via opencv dnn function
        DLTensor* input;
        constexpr int dtype_code = kDLFloat;
        constexpr int dtype_bits = 32;
        constexpr int dtype_lanes = 1;
        constexpr int device_type = kDLCPU;
        constexpr int device_id = 0;
        constexpr int in_ndim = 4;
        const int64_t in_shape[in_ndim] = {1, 3, 112, 112};
        TVMArrayAlloc(in_shape, in_ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &input);//
        TVMArrayCopyFromBytes(input,tensor.data,112*3*112*4);
        tvm::runtime::Module* mod = (tvm::runtime::Module*)handle;
        tvm::runtime::PackedFunc set_input = mod->GetFunction("set_input");
        set_input("data", input);
        tvm::runtime::PackedFunc run = mod->GetFunction("run");
        run();
        tvm::runtime::PackedFunc get_output = mod->GetFunction("get_output");
        tvm::runtime::NDArray res = get_output(0);
        cv::Mat vector(512,1,CV_32F);
        memcpy(vector.data,res->data,512*4);
        cv::Mat _l2;
        cv::multiply(vector,vector,_l2);
        float l2 = cv::sqrt(cv::sum(_l2).val[0]);
        vector = vector / l2;
        TVMArrayFree(input);
        return vector;
    }
};

inline float CosineDistance(const cv::Mat &v1,const cv::Mat &v2){
    return static_cast<float>(v1.dot(v2));
}


cv::Mat getTemplate(const std::string& imagePath, FR_MFN_Deploy& deploy) {
    cv::Mat data = cv::imread(imagePath);
    auto time_1 = Clock::now();
    cv::Mat out = deploy.forward(data);
    auto time_2 = Clock::now();
    std::cout<<"<spend time>" << std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(time_2 - time_1).count()) << std::endl;
    return out;
}

std::vector<std::string> getstr(std::string line0)
{
    std::vector<int> index1;
    for(int i=0;i<line0.size();i++)
    {
        if(line0[i]==' ')index1.emplace_back(i);
    }
    std::cout<<"count \\n "<<index1.size()<<std::endl;
    std::vector<std::string> outstr;
    outstr.emplace_back(line0.substr(0, index1[0]));
    outstr.emplace_back(line0.substr(index1[0]+1, index1[1]-index1[0]-1));
    outstr.emplace_back(line0.substr(index1[1]+1));
    return outstr;
}

int main() {
    std::cout << "Loading the model" << std::endl;
    FR_MFN_Deploy deploy("./");
    std::cout << "Loaded model" << std::endl;

// Same person
    std::vector<std::string> imagePaths = {"t2.jpg"};
    std::ifstream infile("./multi/multi.lst");
    std::string line;
    cv::Mat vec1,vec2;
    unsigned int count_all=0;
    unsigned int true_cou=0;
    char flag;
    while(getline(infile, line)){
        std::vector<std::string> outstr = getstr(line);
        vec1 = getTemplate(outstr[0], deploy);
        vec2 = getTemplate(outstr[1], deploy);
        float score = vec1.dot(vec2);
        if(score>0.6){
            flag = '1';
        }
        else flag = '0';
        if(flag==outstr[2][0]) true_cou++;
        count_all++;
        std::cout<<"similarity "<<score<<" out string: "<<outstr[0]<<" "<<outstr[1]<<" "<<outstr[2]<<std::endl;
    }
    float rate = true_cou /(float)count_all;
    std::cout<<typeid(rate).name()<<std::endl;
    std::cout<<"[rate] "<<rate<<" [true]"<<true_cou<<" [all]"<<count_all<<std::endl;

    return 0;
}
