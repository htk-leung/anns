// #pragma once

// #include "hnswalg.h"

// #include <map> // EDIT
// #include <utility> // EDIT

// namespace hnswlib {

//     // EDIT : hnsw constructor for when used with hnswM
//     HierarchicalNSW::HierarchicalNSW(
//         HierarchicalNSW::SpaceInterface<dist_t> *s,
//         size_t max_elements,
//         size_t M = 16,
//         size_t ef_construction = 200,
//         size_t random_seed = 100,
//         std::map<int, dist_t>* cache,
//         bool allow_replace_deleted = false)
//         : dist_cache_(cache), // EDIT
//             label_op_locks_(MAX_LABEL_OPERATION_LOCKS),
//             link_list_locks_(max_elements),
//             element_levels_(max_elements),
//             allow_replace_deleted_(allow_replace_deleted) 
//     {
//         max_elements_ = max_elements;
//         num_deleted_ = 0;
//         data_size_ = s->get_data_size();
//         fstdistfunc_ = s->get_dist_func();
//         dist_func_param_ = s->get_dist_func_param();
//         if ( M <= 10000 ) {
//             M_ = M;
//         } else {
//             HNSWERR << "warning: M parameter exceeds 10000 which may lead to adverse effects." << std::endl;
//             HNSWERR << "         Cap to 10000 will be applied for the rest of the processing." << std::endl;
//             M_ = 10000;
//         }
//         maxM_ = M_;
//         maxM0_ = M_ * 2;
//         ef_construction_ = std::max(ef_construction, M_);
//         ef_ = 10;

//         level_generator_.seed(random_seed);
//         update_probability_generator_.seed(random_seed + 1);

//         size_links_level0_ = maxM0_ * sizeof(tableint) + sizeof(linklistsizeint);
//         size_data_per_element_ = size_links_level0_ + data_size_ + sizeof(labeltype);
//         offsetData_ = size_links_level0_;
//         label_offset_ = size_links_level0_ + data_size_;
//         offsetLevel0_ = 0;

//         data_level0_memory_ = (char *) malloc(max_elements_ * size_data_per_element_);
//         if (data_level0_memory_ == nullptr)
//             throw std::runtime_error("Not enough memory");

//         cur_element_count = 0;

//         visited_list_pool_ = std::unique_ptr<VisitedListPool>(new VisitedListPool(1, max_elements));

//         // initializations for special treatment of the first node
//         enterpoint_node_ = -1;
//         maxlevel_ = -1;

//         linkLists_ = (char **) malloc(sizeof(void *) * max_elements_);
//         if (linkLists_ == nullptr)
//             throw std::runtime_error("Not enough memory: HierarchicalNSW failed to allocate linklists");
//         size_links_per_element_ = maxM_ * sizeof(tableint) + sizeof(linklistsizeint);
//         mult_ = 1 / log(1.0 * M_);
//         revSize_ = 1.0 / mult_;
//     }

//     template<typename dist_t>
//     inline dist_t 
//     HierarchicalNSW::findDistWithCache(tableint nodeId, const void *data_point, char* ep_data) 
//     {
//         // return if found in cache
//         auto inCache = dist_cache_.find();
//         if (inCache != dist_cache_.end())
//             return (*dist_cache_)[cand];

//         // else calculate, save to cache and return
//         metric_distance_computations++;
//         dist_t d = fstdistfunc_(data_point, ep_data, dist_func_param_);
//         dist_cache_->insert({cand, d});
//         return d;
//     }

//     // bare_bone_search means there is no check for deletions and stop condition is ignored in return of extra performance
//     template <bool bare_bone_search = true, bool collect_metrics = false>
//     std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
//     HierarchicalNSW::searchBaseLayerSTMC(
//         tableint ep_id,
//         const void *data_point,
//         size_t ef,
//         int dc_budget,
//         BaseFilterFunctor* isIdAllowed = nullptr,
//         BaseSearchStopCondition<dist_t>* stop_condition = nullptr) const 
//     {

//         VisitedList *vl = visited_list_pool_->getFreeVisitedList(); // list to check if already visited IN THIS LAYER
//         vl_type *visited_array = vl->mass;
//         vl_type visited_array_tag = vl->curV;

//         std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
//         std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidate_set;

//         dist_t lowerBound;
//         if (bare_bone_search || 
//             (!isMarkedDeleted(ep_id) && ((!isIdAllowed) || (*isIdAllowed)(getExternalLabel(ep_id))))) {
//             char* ep_data = getDataByInternalId(ep_id);

//             // EDIT : cache
//             dist_t dist = findDistWithCache<dist_t>(cand, data_point, ep_data);

//             lowerBound = dist;
//             top_candidates.emplace(dist, ep_id);
//             if (!bare_bone_search && stop_condition) {
//                 stop_condition->add_point_to_result(getExternalLabel(ep_id), ep_data, dist);
//             }
//             candidate_set.emplace(-dist, ep_id);
//         } else {
//             lowerBound = std::numeric_limits<dist_t>::max();
//             candidate_set.emplace(-lowerBound, ep_id);
//         }

//         visited_array[ep_id] = visited_array_tag;

//         while (!candidate_set.empty() && metric_distance_computations <= dc_budget) {

//             metric_hops++;

//             std::pair<dist_t, tableint> current_node_pair = candidate_set.top();
//             dist_t candidate_dist = -current_node_pair.first;

//             bool flag_stop_search;
//             if (bare_bone_search) {
//                 flag_stop_search = candidate_dist > lowerBound;
//             } else {
//                 if (stop_condition) {
//                     flag_stop_search = stop_condition->should_stop_search(candidate_dist, lowerBound);
//                 } else {
//                     flag_stop_search = candidate_dist > lowerBound && top_candidates.size() == ef;
//                 }
//             }
//             if (flag_stop_search) {
//                 break;
//             }
//             candidate_set.pop();

//             tableint current_node_id = current_node_pair.second;
//             int *data = (int *) get_linklist0(current_node_id);
//             size_t size = getListCount((linklistsizeint*)data);
//             //  bool cur_node_deleted = isMarkedDeleted(current_node_id);


//             #ifdef USE_SSE
//                 _mm_prefetch((char *) (visited_array + *(data + 1)), _MM_HINT_T0);
//                 _mm_prefetch((char *) (visited_array + *(data + 1) + 64), _MM_HINT_T0);
//                 _mm_prefetch(data_level0_memory_ + (*(data + 1)) * size_data_per_element_ + offsetData_, _MM_HINT_T0);
//                 _mm_prefetch((char *) (data + 2), _MM_HINT_T0);
//             #endif

//             //  explore neighbors to find candidate
//             for (size_t j = 1; j <= size; j++) {
//                 int candidate_id = *(data + j);
//                 //      if (candidate_id == 0) continue;

//                 #ifdef USE_SSE
//                     _mm_prefetch((char *) (visited_array + *(data + j + 1)), _MM_HINT_T0);
//                     _mm_prefetch(data_level0_memory_ + (*(data + j + 1)) * size_data_per_element_ + offsetData_,
//                                     _MM_HINT_T0);  ////////////
//                 #endif

//                 if (!(visited_array[candidate_id] == visited_array_tag)) {
//                     visited_array[candidate_id] = visited_array_tag;

//                     char *currObj1 = (getDataByInternalId(candidate_id));

//                     // EDIT : cache
//                     dist_t dist = findDistWithCache<dist_t>(candidate_id, data_point, currObj1);

//                     bool flag_consider_candidate;
//                     if (!bare_bone_search && stop_condition) {
//                         flag_consider_candidate = stop_condition->should_consider_candidate(dist, lowerBound);
//                     } else {
//                         flag_consider_candidate = top_candidates.size() < ef || lowerBound > dist;
//                     }

//                     if (flag_consider_candidate) {
//                         candidate_set.emplace(-dist, candidate_id);
//         #ifdef USE_SSE
//                         _mm_prefetch(data_level0_memory_ + candidate_set.top().second * size_data_per_element_ +
//                                         offsetLevel0_,  ///////////
//                                         _MM_HINT_T0);  ////////////////////////
//         #endif

//                         if (bare_bone_search || 
//                             (!isMarkedDeleted(candidate_id) && ((!isIdAllowed) || (*isIdAllowed)(getExternalLabel(candidate_id))))) {
//                             top_candidates.emplace(dist, candidate_id);
//                             if (!bare_bone_search && stop_condition) {
//                                 stop_condition->add_point_to_result(getExternalLabel(candidate_id), currObj1, dist);
//                             }
//                         }

//                         bool flag_remove_extra = false;
//                         if (!bare_bone_search && stop_condition) {
//                             flag_remove_extra = stop_condition->should_remove_extra();
//                         } else {
//                             flag_remove_extra = top_candidates.size() > ef;
//                         }
//                         while (flag_remove_extra) {
//                             tableint id = top_candidates.top().second;
//                             top_candidates.pop();
//                             if (!bare_bone_search && stop_condition) {
//                                 stop_condition->remove_point_from_result(getExternalLabel(id), getDataByInternalId(id), dist);
//                                 flag_remove_extra = stop_condition->should_remove_extra();
//                             } else {
//                                 flag_remove_extra = top_candidates.size() > ef;
//                             }
//                         }

//                         if (!top_candidates.empty())
//                             lowerBound = top_candidates.top().first;
//                     }
//                 }
//             }
//         }

//         visited_list_pool_->releaseVisitedList(vl);
//         // EDIT : stop timer
//         // const std::chrono::time_point<std::chrono::high_resolution_clock> t2 = std::chrono::high_resolution_clock::now();
//         // EDIT : add time to variable
//         // metric_latency_base = metric_latency_base + std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000.0;

//         return top_candidates;
//     }

//     // EDIT : knnsearch function to use distance caching 
//     std::priority_queue<std::pair<dist_t, labeltype >>
//     HierarchicalNSW::searchKnnMC(int dc_budget,
//         const void *query_data, 
//         size_t k, 
//         BaseFilterFunctor* isIdAllowed = nullptr) const 
//     {

//         std::priority_queue<std::pair<dist_t, labeltype >> result;
//         if (cur_element_count == 0) return result;

//         tableint currObj = enterpoint_node_;
//         dist_t curdist = fstdistfunc_(query_data, getDataByInternalId(enterpoint_node_), dist_func_param_);

//         // dist comp count
//         metric_distance_computations++;

//         // EDIT save dist to cache
//         dist_cache_->insert({currObj, curdist});

//         // at each level
//         for (int level = maxlevel_; level > 0; level--) {
//             bool changed = true;

//             // for each node hopped to find nearest neighbors until selection doesn't change
//             while (changed && metric_distance_computations <= dc_budget) { 
//             // EDIT : budget condition, but may be redundant since this step should always be within budget?? 
//             //          well not if you have many many graphs
//                 changed = false;
//                 unsigned int *data;

//                 // get neighbors
//                 data = (unsigned int *) get_linklist(currObj, level);
//                 int size = getListCount(data);

//                 // update metrics
//                 metric_hops++;
//                 // metric_distance_computations+=size;

//                 // datal = neighbors list
//                 tableint *datal = (tableint *) (data + 1);

//                 // pick closest neighbor
//                 for (int i = 0; i < size; i++) {

//                     tableint cand = datal[i];
//                     if (cand < 0 || cand > max_elements_)
//                         throw std::runtime_error("cand error");

//                     // EDIT : distance computation / cache
//                     // if found in cache, use cache
//                     dist_t d;
//                     auto inCache = dist_cache_.find();
//                     if (inCache != dist_cache_.end())
//                         d = (*dist_cache_)[cand];
//                     // else calculate and save to cache
//                     else {
//                         metric_distance_computations++;
//                         d = fstdistfunc_(query_data, getDataByInternalId(cand), dist_func_param_);
//                         dist_cache_->insert({cand, d});
//                     }

//                     // greedy search : select if closer to target
//                     if (d < curdist) {
//                         curdist = d;
//                         currObj = cand;
//                         changed = true;
//                     }
//                 }
//             }
//         }

//         // found entry point to base layer by now

//         // base layer search setup
//         std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
//         bool bare_bone_search = !num_deleted_ && !isIdAllowed;

//         if (bare_bone_search) {
//             top_candidates = searchBaseLayerSTMC<true, true>(
//                     currObj, query_data, std::max(ef_, k), isIdAllowed);
//         } else {
//             top_candidates = searchBaseLayerSTMC<false>(
//                     currObj, query_data, std::max(ef_, k), isIdAllowed);
//         }

//         // trim extra
//         while (top_candidates.size() > k) {
//             top_candidates.pop();
//         }
//         // save to result >> max first or min first???
//         while (top_candidates.size() > 0) {
//             std::pair<dist_t, tableint> rez = top_candidates.top();
//             result.push(std::pair<dist_t, labeltype>(rez.first, getExternalLabel(rez.second)));
//             top_candidates.pop();
//         }

//         return result;
//     }

//     // EDIT : base layer beam search for multiple graph with distance computation budget conditions
//     template <bool bare_bone_search = true, bool collect_metrics = false>
//     std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
//     HierarchicalNSW::searchBaseLayerSTM(
//         tableint ep_id,
//         const void *data_point,
//         size_t ef,
//         int dc_budget,
//         BaseFilterFunctor* isIdAllowed = nullptr,
//         BaseSearchStopCondition<dist_t>* stop_condition = nullptr) const 
//     {

//         VisitedList *vl = visited_list_pool_->getFreeVisitedList();
//         vl_type *visited_array = vl->mass;
//         vl_type visited_array_tag = vl->curV;

//         std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
//         std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidate_set;

//         // entry node
//         dist_t lowerBound;
//         if (bare_bone_search || 
//             (!isMarkedDeleted(ep_id) && ((!isIdAllowed) || (*isIdAllowed)(getExternalLabel(ep_id))))) {

//             // base layer entry node
//             char* ep_data = getDataByInternalId(ep_id);

//             dist_t dist = fstdistfunc_(data_point, ep_data, dist_func_param_);
//             metric_distance_computations++;

//             lowerBound = dist;
//             top_candidates.emplace(dist, ep_id);
//             if (!bare_bone_search && stop_condition) {
//                 stop_condition->add_point_to_result(getExternalLabel(ep_id), ep_data, dist);
//             }
//             candidate_set.emplace(-dist, ep_id);
//         } else {
//             lowerBound = std::numeric_limits<dist_t>::max();
//             candidate_set.emplace(-lowerBound, ep_id);
//         }

//         visited_array[ep_id] = visited_array_tag;

//         // beam search through candidates list
//         while (!candidate_set.empty() && metric_distance_computations <= dc_budget) { // EDIT : added dc_budget condition
            
//             metric_hops++;

//             std::pair<dist_t, tableint> current_node_pair = candidate_set.top();
//             dist_t candidate_dist = -current_node_pair.first;

//             bool flag_stop_search;
//             if (bare_bone_search) {
//                 flag_stop_search = candidate_dist > lowerBound;
//             } else {
//                 if (stop_condition) {
//                     flag_stop_search = stop_condition->should_stop_search(candidate_dist, lowerBound);
//                 } else {
//                     flag_stop_search = candidate_dist > lowerBound && top_candidates.size() == ef;
//                 }
//             }
//             if (flag_stop_search) {
//                 break;
//             }
//             candidate_set.pop();

//             tableint current_node_id = current_node_pair.second;
//             int *data = (int *) get_linklist0(current_node_id);
//             size_t size = getListCount((linklistsizeint*)data);
//             //  bool cur_node_deleted = isMarkedDeleted(current_node_id);

//             //     metric_hops++;
//             //     metric_distance_computations+=size; // << wrong place??

//         #ifdef USE_SSE
//             _mm_prefetch((char *) (visited_array + *(data + 1)), _MM_HINT_T0);
//             _mm_prefetch((char *) (visited_array + *(data + 1) + 64), _MM_HINT_T0);
//             _mm_prefetch(data_level0_memory_ + (*(data + 1)) * size_data_per_element_ + offsetData_, _MM_HINT_T0);
//             _mm_prefetch((char *) (data + 2), _MM_HINT_T0);
//         #endif
//             // explore neighbors to find candidate
//             for (size_t j = 1; j <= size; j++) {
//                 int candidate_id = *(data + j);
//         //  if (candidate_id == 0) continue;
//         #ifdef USE_SSE
//                 _mm_prefetch((char *) (visited_array + *(data + j + 1)), _MM_HINT_T0);
//                 _mm_prefetch(data_level0_memory_ + (*(data + j + 1)) * size_data_per_element_ + offsetData_,
//                                 _MM_HINT_T0);  ////////////
//         #endif
//                 if (!(visited_array[candidate_id] == visited_array_tag)) {
//                     visited_array[candidate_id] = visited_array_tag;

//                     char *currObj1 = (getDataByInternalId(candidate_id));

//                     dist_t dist = fstdistfunc_(data_point, currObj1, dist_func_param_);
//                     // EDIT : moved increment here
//                     metric_distance_computations++;

//                     bool flag_consider_candidate;
//                     if (!bare_bone_search && stop_condition) {
//                         flag_consider_candidate = stop_condition->should_consider_candidate(dist, lowerBound);
//                     } else {
//                         flag_consider_candidate = top_candidates.size() < ef || lowerBound > dist;
//                     }

//                     if (flag_consider_candidate) {
//                         candidate_set.emplace(-dist, candidate_id);
//         #ifdef USE_SSE
//                         _mm_prefetch(data_level0_memory_ + candidate_set.top().second * size_data_per_element_ +
//                                         offsetLevel0_,  ///////////
//                                         _MM_HINT_T0);  ////////////////////////
//         #endif

//                         if (bare_bone_search || 
//                             (!isMarkedDeleted(candidate_id) && ((!isIdAllowed) || (*isIdAllowed)(getExternalLabel(candidate_id))))) {
//                             top_candidates.emplace(dist, candidate_id);
//                             if (!bare_bone_search && stop_condition) {
//                                 stop_condition->add_point_to_result(getExternalLabel(candidate_id), currObj1, dist);
//                             }
//                         }

//                         bool flag_remove_extra = false;
//                         if (!bare_bone_search && stop_condition) {
//                             flag_remove_extra = stop_condition->should_remove_extra();
//                         } else {
//                             flag_remove_extra = top_candidates.size() > ef;
//                         }
//                         while (flag_remove_extra) {
//                             tableint id = top_candidates.top().second;
//                             top_candidates.pop();
//                             if (!bare_bone_search && stop_condition) {
//                                 stop_condition->remove_point_from_result(getExternalLabel(id), getDataByInternalId(id), dist);
//                                 flag_remove_extra = stop_condition->should_remove_extra();
//                             } else {
//                                 flag_remove_extra = top_candidates.size() > ef;
//                             }
//                         }

//                         if (!top_candidates.empty())
//                             lowerBound = top_candidates.top().first;
//                     }
//                 }
//             }
//         }

//         visited_list_pool_->releaseVisitedList(vl);
//         // EDIT : stop timer
//         // const std::chrono::time_point<std::chrono::high_resolution_clock> t2 = std::chrono::high_resolution_clock::now();
//         // EDIT : add time to variable
//         // metric_latency_base = metric_latency_base + std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000.0;

//         return top_candidates;
//     }

//     // EDIT : upper layer greedy search for multiple graph with distance computation budget conditions
//     std::priority_queue<std::pair<dist_t, labeltype >>
//     HierarchicalNSW::searchKnnM(int dc_budget,
//         const void *query_data, size_t k, BaseFilterFunctor* isIdAllowed = nullptr) const 
//     {

//         std::priority_queue<std::pair<dist_t, labeltype >> result;
//         if (cur_element_count == 0) return result;

//         // EDIT : start timer
//         // const std::chrono::time_point<std::chrono::high_resolution_clock> t1 = std::chrono::high_resolution_clock::now();

//         tableint currObj = enterpoint_node_;
//         dist_t curdist = fstdistfunc_(query_data, getDataByInternalId(enterpoint_node_), dist_func_param_);

//         // EDIT : add layer traversal dist comp count
//         metric_distance_computations++;

//         for (int level = maxlevel_; level > 0; level--) {
//             bool changed = true;
//             while (changed && metric_distance_computations <= dc_budget) { // EDIT : changed stopping condition for loop
//                 changed = false;
//                 unsigned int *data;

//                 data = (unsigned int *) get_linklist(currObj, level);
//                 int size = getListCount(data);

//                 metric_hops++;
//                 metric_distance_computations+=size;

//                 tableint *datal = (tableint *) (data + 1);
//                 for (int i = 0; i < size; i++) {
//                     tableint cand = datal[i];
//                     if (cand < 0 || cand > max_elements_)
//                         throw std::runtime_error("cand error");
//                     dist_t d = fstdistfunc_(query_data, getDataByInternalId(cand), dist_func_param_);

//                     if (d < curdist) {
//                         curdist = d;
//                         currObj = cand;
//                         changed = true;
//                     }
//                 }
//             }
//         }

//         std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
//         bool bare_bone_search = !num_deleted_ && !isIdAllowed;

//         // EDIT : stop timer
//         // const std::chrono::time_point<std::chrono::high_resolution_clock> t2 = std::chrono::high_resolution_clock::now();

//         if (bare_bone_search) {
//             // EDIT : added 2nd parameter=true for function call to enable metric collection
//             top_candidates = searchBaseLayerSTM<true, true>(
//                     currObj, query_data, std::max(ef_, k), dc_budget, isIdAllowed);
//         } else {
//             top_candidates = searchBaseLayerSTM<false>(
//                     currObj, query_data, std::max(ef_, k), dc_budget, isIdAllowed);
//         }

//         // EDIT : start timer
//         // const std::chrono::time_point<std::chrono::high_resolution_clock> t3 = std::chrono::high_resolution_clock::now();

//         while (top_candidates.size() > k) {
//             top_candidates.pop();
//         }
//         while (top_candidates.size() > 0) {
//             std::pair<dist_t, tableint> rez = top_candidates.top();
//             result.push(std::pair<dist_t, labeltype>(rez.first, getExternalLabel(rez.second)));
//             top_candidates.pop();
//         }

//         // EDIT : stop timer
//         // const std::chrono::time_point<std::chrono::high_resolution_clock> t4 = std::chrono::high_resolution_clock::now();
//         // EDIT : save time to layer traversal var
//         // metric_latency_upper = metric_latency_upper + std::chrono::duration_cast<std::chrono::microseconds>((t2 - t1) + (t4 - t3)).count() / 1000.0;

//         return result;
//     }

// }  // namespace hnswlib