/*
 *LinearFold.h*
 header file for LinearFold.cpp.

 author: Kai Zhao, Dezhong Deng, He Zhang, Liang Zhang
 edited by: 03/2021
*/
#pragma once

#ifndef FASTCKY_BEAMCKYPAR_H
#define FASTCKY_BEAMCKYPAR_H

#include <string>
#include <limits>
#include <vector>
#include <unordered_map>
#include "../fold.h"

// #define is_cube_pruning
#define is_candidate_list

#define MIN_CUBE_PRUNING_SIZE 20
#define FOR_MXFOLD2
//#define lv

namespace LinearFold {

#ifdef FOR_MXFOLD2
    typedef float value_type;
#elif defined(lv)
    typedef int value_type;
#else
    typedef double value_type;
#endif
#define VALUE_MIN std::numeric_limits<value_type>::lowest()

enum Manner {
  MANNER_NONE = 0,              // 0: empty
  MANNER_H,                     // 1: hairpin candidate
  MANNER_HAIRPIN,               // 2: hairpin
  MANNER_SINGLE,                // 3: single
  MANNER_HELIX,                 // 4: helix
  MANNER_MULTI,                 // 5: multi = ..M2. [30 restriction on the left and jump on the right]
  MANNER_MULTI_eq_MULTI_plus_U, // 6: multi = multi + U
  MANNER_P_eq_MULTI,            // 7: P = (multi)
  MANNER_M2_eq_M_plus_P,        // 8: M2 = M + P
  MANNER_M_eq_M2,               // 9: M = M2
  MANNER_M_eq_M_plus_U,         // 10: M = M + U
  MANNER_M_eq_P,                // 11: M = P
  MANNER_C_eq_C_plus_U,     // 12: C = C + U
  MANNER_C_eq_C_plus_P,     // 13: C = C + P
};

enum BestTypes {
  TYPE_C = 0,
  TYPE_H,
  TYPE_P,
  TYPE_M,
  TYPE_Multi,
  TYPE_M2,
};

struct State {
    // double score;
    value_type score;
    Manner manner;

    union TraceInfo {
        int split;
        struct {
            int l1;
            int l2;
        } paddings;
    };

    TraceInfo trace;

    State(): manner(MANNER_NONE), score(VALUE_MIN) {};
    State(value_type s, Manner m): score(s), manner(m) {};

    void set(value_type score_, Manner manner_) {
        score = score_; manner = manner_;
    }

    void set(value_type score_, Manner manner_, int split_) {
        score = score_; manner = manner_; trace.split = split_;
    }

    void set(value_type score_, Manner manner_, int l1_, int l2_) {
        score = score_; manner = manner_;
        trace.paddings.l1 = l1_; trace.paddings.l2 = l2_;
    }
};


// constraint
enum {
    C_ANY       = -1, // '.'
    C_UNPAIRED  = -2, // 'x'
    C_PAIRED_L  = -3, // '<'
    C_PAIRED_R  = -4, // '>'
    C_PAIRED_LR = -5, // '|'
};

template < typename P, typename S = typename P::ScoreType >
class BeamCKYParser : public Fold {
public:
    struct Options : public Fold::Options
    {
        std::vector<int> cons;
        Options& constraints(const std::vector<int>& c)
        {
            this->cons = c;
            return *this;
        }
    };

    std::unique_ptr<P> param_;
    int beam;
    //int min_hairpin_loop;
    //int max_single_loop;
    std::pair<value_type, value_type> loss_paired_;
    bool is_verbose;
    bool use_constraints; // lisiz, add constraints
    //bool zuker;
    //int  window_size; //2 + 1 + 2 = 5 in total, 5*5 window size.
    //float zuker_energy_delta;
    //bool use_shape = false;
    //double m = 1.8;
    //double b = -0.6;
    //bool is_fasta = false;

    struct DecoderResult {
        //std::string structure;
        value_type score;
        unsigned long num_states;
        double time;
    };

    BeamCKYParser(
                  std::unique_ptr<P>&& p, 
                  int beam_size=100
                  //bool is_verbose=false,
                  //bool is_constraints=false,
                  //bool zuker_subopt=false,
                  //float zuker_energy_delta=5.0,
                  //std::string shape_file_path="",
                  //bool is_fasta=false
                  ); // lisiz, add constraints

    auto parse(const std::string& seq, const Options& opts = Options()) -> DecoderResult;
#if 0
    void outside(std::vector<int> next_pair[]); //for zuker subopt
#endif
    auto traceback(const std::string& seq, const Options& opts = Options()) -> std::pair<value_type, std::vector<uint32_t>>;

private:
#if 0
    void get_parentheses(char* result, const std::string& seq);

    std::pair<std::string, std::string> get_parentheses_outside_real_backtrace(int i, int j, State& state_beta, std::map<std::tuple<BestTypes, int, int>, std::pair<std::string, std::string> >& global_visited_outside, std::map<std::tuple<BestTypes, int, int>, std::string>& global_visited_inside, std::set<std::pair<int,int> >& window_visited);
    std::string get_parentheses_inside_real_backtrace(int i, int j, State& state, std::map<std::tuple<BestTypes, int, int>, std::string>& global_visited_inside, std::set<std::pair<int,int> >& window_visited);
#endif

    unsigned seq_length;

    std::vector<std::unordered_map<int, State>> bestH, bestP, bestM2, bestMulti, bestM;

    //Zuker subopt
    std::vector<std::unordered_map<int, State>> bestH_beta, bestP_beta, bestM2_beta, bestMulti_beta, bestM_beta;

    std::vector<int> if_tetraloops;
    std::vector<int> if_hexaloops;
    std::vector<int> if_triloops;

    // same as bestM, but ordered
    std::vector<std::vector<std::pair<value_type, int>>> sorted_bestM;

    // hzhang: sort keys in each beam to avoid randomness
    std::vector<std::pair<int, State>> keys;

    // hzhang: sort keys in each beam to avoid randomness
    void sort_keys(std::unordered_map<int, State> &map, std::vector<std::pair<int,State>> &sorted_keys);

    void sortM(value_type threshold,
               std::unordered_map<int, State> &beamstep,
               std::vector<std::pair<value_type, int>>& sorted_stepM);
    std::vector<State> bestC;
    std::vector<int> nucs;

    //Zuker subopt
    std::vector<State> bestC_beta;
    
    // SHAPE
    // std::vector<double> SHAPE_data;
    // std::vector<int> pseudo_energy_stack;


    // lisiz: constraints
    std::vector<int> allow_unpaired_position;
    std::vector<int> allow_unpaired_range;
    bool allow_paired(int i, int j, const std::vector<int>* cons, char nuci, char nucj);

    void prepare(unsigned len);

    void update_if_better(State &state, value_type newscore, Manner manner) {
      if (state.score < newscore)
            state.set(newscore, manner);
    };

    void update_if_better(State &state, value_type newscore, Manner manner, int split) {
        if (state.score < newscore || state.manner == MANNER_NONE)
            state.set(newscore, manner, split);
    };

    void update_if_better(State &state, value_type newscore, Manner manner, int l1, int l2) {
        assert(l1>=0); assert(l2>=0);
        if (state.score < newscore || state.manner == MANNER_NONE)
            state.set(newscore, manner, l1, l2);
    };

    value_type beam_prune(std::unordered_map<int, State>& beamstep);

    // vector to store the scores at each beam temporarily for beam pruning
    std::vector<std::pair<value_type, int>> scores;
};

};

#endif //FASTCKY_BEAMCKYPAR_H
