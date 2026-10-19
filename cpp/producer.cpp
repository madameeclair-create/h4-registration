// 【继承MAC部分】候选生成及记录适配；不是本文创新模块。
// 二阶图、极大团、节点引导和候选SVD沿用MAC；新增选择见h4_registration/core_indoor.py。
// Descriptor-selectable adapter; upstream registration implementation unchanged.
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <iostream>
#include <fstream>
#include <mutex>
#include <array>
#include <cstdint>
#include <algorithm>
#include <iomanip>
#include <limits>
#include <chrono>
#include "Eva.h"
bool add_overlap = false;
bool low_inlieratio = false;
bool no_logs = false;
std::string folderPath;

std::mutex cache_mutex;
struct Record {Eigen::Matrix4d pose; std::vector<std::array<float,6>> clique;};
std::vector<Record> cache_hypotheses;
void mac_record_hypothesis(const Eigen::Matrix4d& pose,const pcl::PointCloud<pcl::PointXYZ>& s,const pcl::PointCloud<pcl::PointXYZ>& t) {
    Record r;r.pose=pose;
    for(size_t i=0;i<s.size();++i)r.clique.push_back({s[i].x,s[i].y,s[i].z,t[i].x,t[i].y,t[i].z});
    std::sort(r.clique.begin(),r.clique.end());
    std::lock_guard<std::mutex> guard(cache_mutex);cache_hypotheses.push_back(std::move(r));
}
void save_hypotheses(const char* filename) {
    std::stable_sort(cache_hypotheses.begin(),cache_hypotheses.end(),[](const Record& a,const Record& b){
        if(a.pose.allFinite()!=b.pose.allFinite())return a.pose.allFinite();
        if(!a.pose.allFinite())return a.clique<b.clique;
        for(int r=0;r<4;++r)for(int c=0;c<4;++c){if(a.pose(r,c)<b.pose(r,c))return true;if(a.pose(r,c)>b.pose(r,c))return false;}
        return a.clique<b.clique;
    });
    std::ofstream f(filename,std::ios::binary);f.write("MACPRV01",8);
    uint64_t n=cache_hypotheses.size();f.write(reinterpret_cast<char*>(&n),8);
    for(auto& r:cache_hypotheses){
        for(int a=0;a<4;++a)for(int b=0;b<4;++b)f.write(reinterpret_cast<char*>(&r.pose(a,b)),8);
        uint64_t m=r.clique.size();f.write(reinterpret_cast<char*>(&m),8);
        for(auto& pair:r.clique)for(float value:pair)f.write(reinterpret_cast<char*>(&value),4);
    }
    if(!f)throw std::runtime_error("provenance write failed");
    std::cout<<"MAC_PROVENANCE "<<n<<std::endl;
}

int main(int argc, char** argv) {
    if (argc != 10) {
        std::cerr << "usage: mac_pair_descriptors dataset source target corr label gt output descriptor hypothesis_cache\n";
        return 2;
    }
    std::string descriptor = argv[8];
    if (descriptor != "fpfh" && descriptor != "fcgf") return 2;
    double re = std::numeric_limits<double>::quiet_NaN(), te = re;
    double inliers=0, total=0, ratio=0, successful_hypotheses=0, hypothesis_count=0;
    std::vector<double> timing;
    folderPath = argv[7];
    igraph_rng_seed(igraph_rng_default(), 123);
    std::cout << "MAC_RNG_SEED 123\n";
    auto start = std::chrono::steady_clock::now();
    bool success = registration(argv[1], argv[2], argv[3], argv[4], argv[5], "NULL",
        argv[6], argv[7], re, te, inliers, total, ratio, successful_hypotheses,
        hypothesis_count, descriptor, timing);
    auto seconds = std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    std::cout << std::setprecision(17) << "MAC_PAIR_RESULT " << success << " " << re << " " << te
              << " " << inliers << " " << total << " " << ratio << " " << seconds << "\n";
    save_hypotheses(argv[9]);
    return 0;
}
