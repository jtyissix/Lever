"""Cloud-side and edge-side storage utilities for Lever."""

from __future__ import annotations

import json
import math
import os
import pickle
import random
from pathlib import Path

import numpy as np
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from sklearn.neighbors import BallTree
from DataStructure import update_packet,node_info
from Prompts import EXTRACT_TAG_PROMPT
from llms import summary_LLM
from config import TAG_MODEL_PATH, TAG_CACHE_PATH, DOMAIN_KNN_DIR, OUTPUT_ROOT
def bool_generator(probability):
    return 1 if random.random() < probability else 0

def get_interest_domain(keyword_array,save_path):

    lm = summary_LLM(str(TAG_MODEL_PATH))

    messages = [[{"role": "system", "content": TAG_MATCH_PROMPT},
                 {"role": "user", "content": f'Input keywords:{",".join(keyword_array)}'}]]
    output=lm.generate(messages)
    print(output)
    with open(save_path,'wb') as f:
        pickle.dump(output[0],f)
def merge_polling(d_list):
    merged = []
    max_len = len(d_list[0])

    # Merge items in a round-robin order.
    for i in range(max_len):
        for item in d_list:
            merged.append(item[i])

    # Keep the first occurrence of each item.
    seen = set()
    result = []
    for item in merged:
        if item not in seen:
            seen.add(item)
            result.append(item)

    return result
class Edge_VectorStore:
    """Utils for Vectorstore at the Edge"""
    def __init__(self,index_path,map_path,tag_path=None):
        self.index_path=index_path
        self.DB=faiss.read_index(index_path) #load edge HNSW database
        with open(map_path, "rb") as f:
            self.map = pickle.load(  # ignore[pickle]: explicit-opt-in
                f
            )
        #pdb.set_trace()
        self.entry_select_record=[0]*len(self.map)
        self.entry_select_record=np.array(self.entry_select_record)
        self.flag=False #flag to True if database is modified
        self.offset=faiss.vector_to_array(self.DB.hnsw.offsets)
        self.neighbor = faiss.vector_to_array(self.DB.hnsw.neighbors)
        if tag_path is not None:
            with open(tag_path,'rb') as f:
                self.id_tag=pickle.load(f)
    def __del__(self):
        if self.flag:
            faiss.write_index(self.DB,self.index_path)
    def search_for_entry_point(self,embedded_query):
        """function to process query from edge
           make sure that embedded_query is 2D np array of float32
           return:
           1.entry point's distance
           2.entry point's id"""
        _,ids=self.DB.search(embedded_query,1)
        self.entry_select_record[ids[0][0]]+=1
        return _,[[self.map[ids[0][0]]]]

    def export_entry_statistics(self, save_path):
        """Persist the current edge-side entry statistics."""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(save_path), self.entry_select_record)
        return save_path

    def load_entry_statistics(self, load_path):
        """Restore edge-side entry statistics from a saved npy file."""
        loaded = np.load(load_path)
        if len(loaded) != len(self.entry_select_record):
            raise ValueError("Loaded entry statistics do not match the edge index size.")
        self.entry_select_record = loaded.astype(self.entry_select_record.dtype, copy=False)
        return self.entry_select_record
    def add_index(self,vectors,indices):#needs to be repaired because no idmap obj now
        """
        :param vectors: np array of 2d of vectors, should be of float32
        :param indices: np array of 1d corresponding cloud indices for vectors. should be of int64
        :return: None
        """
        self.flag=True
        self.DB=faiss.IndexIDMap(self.DB)
        self.DB.add_with_ids(vectors,indices)
    def set_search_params(self,ef_search):
        self.DB.efSearch=ef_search
    def loss_calc(self,alpha):
        hit=np.bincount(self.entry_select_record)
        #loss1=-np.log10((len(self.map)-hit[0])/len(self.map))
        #below is code for loss2,
        loss2=[0]*len(self.entry_select_record)
        loss2=np.array(loss2).astype('float64')
        avg_dists = np.zeros(self.DB.ntotal, dtype=np.float32)
        vectors = np.array([self.DB.reconstruct(i) for i in range(self.DB.ntotal)])
        """Compute a heuristic loss score using the HNSW neighborhood structure."""
        index=self.DB
        hnsw=self.DB.hnsw
        levels = faiss.vector_to_array(hnsw.levels)
        offsets = faiss.vector_to_array(hnsw.offsets)
        neighbors = faiss.vector_to_array(hnsw.neighbors)

        # Iterate over all nodes.
        for node_id in range(index.ntotal):
            # Collect all neighbors of the node.
            start = offsets[node_id]
            end = offsets[node_id + 1]
            node_neighbors = neighbors[start:end].tolist()

            # Remove duplicates and exclude the node itself.
            unique_neighbors = list(set(node_neighbors))
            if node_id in unique_neighbors:
                unique_neighbors.remove(node_id)

            # Compute the average distance to neighbors.
            if len(unique_neighbors) > 0:
                node_vec = vectors[node_id]
                neighbor_vecs = vectors[unique_neighbors]
                dists = np.linalg.norm(node_vec - neighbor_vecs, axis=1)
                avg_dists[node_id] = np.mean(dists)
            else:
                avg_dists[node_id] = 0
        max_dist=np.max(avg_dists)
        min_dist=np.min(avg_dists)
        dist_prob=1
        dist_prob2=(avg_dists-min_dist+1e-1)/(max_dist-min_dist)
        tot=0
        for i in range(len(hit)):
            tot+=hit[i]*i
        '''
        for i in range(len(hit)):
            if i==0:
                continue
            selection=np.where(self.entry_select_record==i)[0]
            for item in selection:
                loss2+=(i/tot)*abs(np.log10((1-i/tot)/dist_prop[item]))
        '''
        for i in range(len(self.entry_select_record)):
            loss2[i]=np.log10((1-self.entry_select_record[i]/tot)/(dist_prob2[i]))
        #return loss1+alpha*loss2  loss1 seemed to be not making sense
        return loss2

    def drop_id_calc(self,drop_rate,keywords):
        """
        0.01 means only keep 1 percent left
        :param drop_rate:
        :param keywords: keyword array
        :return:
        """
        '''
        #calculate to-be-stored zero access index
        num_vec_to_get=[0]*len(self.entry_select_record)
        zero_access_id_temp=np.where(self.entry_select_record==0)[0]
        zero_access_id=[x for x in zero_access_id_temp if self.id_tag[x] not in keywords]
        zero_access_re=[]
        for i in range(len(zero_access_id)):
            if bool_generator(drop_rate):
                zero_access_re.append(zero_access_id[i])
        return np.array(zero_access_re)
        '''
        num_vec_to_get = [0] * len(self.entry_select_record)
        zero_access_id = np.where(self.entry_select_record == 0)[0]
        zero_access_re = []
        for i in range(len(zero_access_id)):
            if bool_generator(drop_rate):
                zero_access_re.append(zero_access_id[i])
        return np.array(zero_access_re)

    def sample_num_calc(self,drop_rate):
        """

        :param drop_rate: for 0 access ids
        :return: array of tuples of cloud-based id for dim 1. num of sampling for this id for dim 2.
        and no_update_array
        """
        hit = np.bincount(self.entry_select_record)
        zero_access_re=self.drop_id_calc(drop_rate)
        no_update_array=[]
        for i in range(len(zero_access_re)):
            no_update_array.append(self.map[zero_access_re[i]])
        assign_num=len(self.entry_select_record)-len(no_update_array)
        tot = 0
        sample_array=[[],[]]
        for i in range(len(hit)):
            tot += hit[i] * i
        for i in range(len(self.entry_select_record)):
            if self.entry_select_record[i]==0:
                continue
            sample_array[0].append(self.map[i])
            sample_array[1].append(math.floor(assign_num*self.entry_select_record[i]/tot))
        return sample_array,no_update_array
    def sample_num_calc_2(self,drop_rate,locality_rate,data_driven_rate,pair,keywords):
        """

        :param drop_rate: for 0 access ids
        :param pair:dict for entry and real ids
        :param locality_rate: for update by locality
        :param data_driven_rate:for update by llm prediction
        :param keywords:LLM prediction of interested Domain
        :return: array of tuples of cloud-based id for dim 1. num of sampling for this id for dim 2.
        and no_update_array
        """
        leaf_num_dict=dict()
        hit = np.bincount(self.entry_select_record)
        zero_access_re=self.drop_id_calc(drop_rate,keywords)
        no_update_array=[]
        for i in range(len(zero_access_re)):
            no_update_array.append(self.map[zero_access_re[i]])
        assign_num=math.ceil(len(self.entry_select_record)-len(no_update_array))#assign_num for locality update
        #assign_num=len(self.entry_select_record)-len(no_update_array)
        tot = 0
        sample_array=[]
        for i in range(len(hit)):
            tot += hit[i] * i
        for i in range(len(self.entry_select_record)):
            if self.entry_select_record[i]==0:
                continue
            node_tot_num=math.ceil(assign_num*self.entry_select_record[i]/tot)
            num=math.ceil(node_tot_num/(1+len(pair[str(self.map[i])])))
            info=node_info(self.map[i],'centroid',num,pair[str(self.map[i])])
            sample_array.append(info)
            for item in pair[str(self.map[i])]:
                if item not in leaf_num_dict.keys():
                    leaf_num_dict[item]=num
                else:
                    leaf_num_dict[item]+=num
        for item in leaf_num_dict.keys():
            info=node_info(int(item),'leaf',leaf_num_dict[item])
            sample_array.append(info)
        data_driven_num=math.ceil((len(self.entry_select_record)-len(no_update_array))*(data_driven_rate/(locality_rate+data_driven_rate)))
        access_id = np.where(self.entry_select_record != 0)[0]
        data_start_point=[x for x in access_id ]
        interest_node_num=math.ceil(data_driven_num/len(data_start_point))
        for i in range(len(self.entry_select_record)):
            if self.entry_select_record[i]==0:
                continue
            interest_node_num=math.ceil(data_driven_num * self.entry_select_record[i] / tot)
            sample_array.append(node_info(self.map[i], 'interest', interest_node_num, pair[str(self.map[i])],
                                          tag=self.id_tag[i]))
        #for item in data_start_point:
            #sample_array.append(node_info(self.map[item],'interest',interest_node_num,pair[str(self.map[item])],tag=self.id_tag[item]))
        #sample_array.append(node_info(-1,'interest',data_driven_num))
        return sample_array,no_update_array
    def sample_num_calc_cover_leaf(self,drop_rate,locality_rate,data_driven_rate,pair,keywords):
        leaf_num_dict = dict()
        hit = np.bincount(self.entry_select_record)
        zero_access_re = self.drop_id_calc(drop_rate, keywords)
        no_update_array = []
        for i in range(len(zero_access_re)):
            no_update_array.append(self.map[zero_access_re[i]])
        # assign_num=math.ceil(len(self.entry_select_record)-len(no_update_array))#assign_num for locality update
        assign_num = math.ceil((len(self.entry_select_record) - len(no_update_array)) * (
                locality_rate / (locality_rate + data_driven_rate)))
        # assign_num=len(self.entry_select_record)-len(no_update_array)
        tot = 0
        sample_array = []
        for i in range(len(hit)):
            tot += hit[i] * i
        for i in range(len(self.entry_select_record)):
            if self.entry_select_record[i] == 0:
                continue
            node_tot_num = math.ceil(assign_num * self.entry_select_record[i] / tot)
            num = math.ceil(node_tot_num / (1 + len(pair[str(self.map[i])])))
            info = node_info(self.map[i], 'centroid', num, pair[str(self.map[i])])
            sample_array.append(info)
            for item in pair[str(self.map[i])]:
                if item not in leaf_num_dict.keys():
                    leaf_num_dict[item] = num
                else:
                    leaf_num_dict[item] += num
        for item in leaf_num_dict.keys():
            info = node_info(int(item), 'leaf', leaf_num_dict[item])
            sample_array.append(info)
        data_driven_num = math.ceil((len(self.entry_select_record) - len(no_update_array)) * (
                    data_driven_rate / (locality_rate + data_driven_rate)))
        access_id = np.where(self.entry_select_record != 0)[0]
        data_start_point = [x for x in access_id]
        interest_node_num = math.ceil(data_driven_num / len(data_start_point))
        for i in range(len(self.entry_select_record)):
            if self.entry_select_record[i] == 0:
                continue
            node_tot_num = math.ceil(data_driven_num * self.entry_select_record[i] / tot)
            num = math.ceil(node_tot_num / (1 + len(pair[str(self.map[i])])))
            sample_array.append(node_info(self.map[i], 'interest', num, pair[str(self.map[i])],
                                          tag=self.id_tag[i]))


            for item in pair[str(self.map[i])]:
                if item not in leaf_num_dict.keys():
                    leaf_num_dict[item] = num
                else:
                    leaf_num_dict[item] += num
            for item in leaf_num_dict.keys():
                info = node_info(int(item), 'leaf', leaf_num_dict[item],tag="leaf_interest")
                sample_array.append(info)
            interest_node_num = math.ceil(data_driven_num * self.entry_select_record[i] / tot)

        # for item in data_start_point:
        # sample_array.append(node_info(self.map[item],'interest',interest_node_num,pair[str(self.map[item])],tag=self.id_tag[item]))
        # sample_array.append(node_info(-1,'interest',data_driven_num))
        return sample_array, no_update_array
    def renew_by_sample(self,sample_ids,sample_vec,faiss_path,map_path):
        if len(sample_ids)<len(self.entry_select_record):
            zero_access_id = np.where(self.entry_select_record == 0)[0]
            filtered_zero_access_id=[]
            for i in range(len(zero_access_id)):
                if  self.map[zero_access_id[i]] not in sample_ids:
                    filtered_zero_access_id.append(zero_access_id[i])
            fit_in_id=random.sample(filtered_zero_access_id,-len(sample_ids)+len(self.entry_select_record))
            for i in range(len(fit_in_id)):
                sample_ids.append(self.map[fit_in_id[i]])
                sample_vec.append(self.DB.reconstruct(int(fit_in_id[i])))
        id_map=[0]*len(sample_ids)
        for i in range(len(sample_ids)):
            id_map[i]=sample_ids[i]
            print(len(id_map))
        index_hnsw = faiss.IndexHNSWFlat(768, 64)  # dim:768,vertex:64
        efSearch = 32  # number of entry points (neighbors) we use on each layer
        efConstruction = 32  # number of entry points used on each layer during construction
        index_hnsw.hnsw.efSearch = efSearch
        index_hnsw.hnsw.efConstruction = efConstruction
        index_hnsw.add(np.array(sample_vec,dtype='float32'))
        faiss.write_index(index_hnsw, faiss_path)
        with open(map_path, "wb") as f:
            pickle.dump(id_map, f)

    def replace_with_cloud_vectors(self, hit_ids, cloud_vecs, cloud_ids,faiss_path,id_path):
        """
        :param hit_ids: list[int] Internal indices selected by the edge side.
        :param cloud_vecs: np.array shape=(k, dim)
        :param cloud_ids: list[int] Cloud-side global IDs aligned with cloud_vecs.
        """

        assert len(cloud_vecs) == len(cloud_ids), "Vectors and IDs must have the same length"

        k = len(cloud_vecs)
        total = self.DB.ntotal

        # Step 1: identify removable nodes.
        all_ids = set(range(total))
        hit_set = set(hit_ids)

        removable = list(all_ids - hit_set)

        if len(removable) < k:
            raise ValueError("Not enough removable nodes.")

        drop_ids = set(random.sample(removable, k))

        # Step 2: rebuild the remaining vectors.
        remain_vecs = []
        remain_map = []
        remain_record = []

        for i in range(total):
            if i in drop_ids:
                continue
            remain_vecs.append(self.DB.reconstruct(i))
            remain_map.append(self.map[i])
            remain_record.append(self.entry_select_record[i])

        # Step 3: append cloud vectors.
        remain_vecs.extend(cloud_vecs)
        remain_map.extend(cloud_ids)
        remain_record.extend([0] * k)

        remain_vecs = np.array(remain_vecs, dtype='float32')

        # Step 4: rebuild the HNSW index.
        dim = remain_vecs.shape[1]

        index_hnsw = faiss.IndexHNSWFlat(768, 64)
        index_hnsw.hnsw.efSearch = 32
        index_hnsw.hnsw.efConstruction = 32

        index_hnsw.add(remain_vecs)

        # Step 5: persist the updated index.
        faiss.write_index(index_hnsw, faiss_path)


        with open(id_path, "wb") as f:
            pickle.dump(remain_map, f)

        print(f"[Edge Update] removed {k}, added {k}, total={index_hnsw.ntotal}")
class Cloud_VectorStore:
    """Utils for Vectorstore at the Cloud"""
    def __init__(self,index_path,text_path,domain_map=None):
        self.index_path=index_path
        self.text_path=text_path
        self.DB=faiss.read_index(index_path) #load edge HNSW database
        with open(text_path, "rb") as f:
            (
                self.docstore,
                self.index_to_docstore_id,
            ) = pickle.load(  # ignore[pickle]: explicit-opt-in
                f
            )
        self.flag=False #flag to True if database is modified
        self.offset = faiss.vector_to_array(self.DB.hnsw.offsets)
        self.neighbor = faiss.vector_to_array(self.DB.hnsw.neighbors)
        self.tag=[]
        if domain_map is not None:
            self.domain_offset=dict()
            self.domain_nb = dict()
            self.domain_knn=dict()
            with open(domain_map,'r',encoding='utf-8') as f:
                self.domain_id_dict=json.load(f)
            for key in self.domain_id_dict.keys():
                self.domain_knn[key]=faiss.read_index(str(DOMAIN_KNN_DIR / f'{key}.faiss'))
                self.domain_offset[key]=faiss.vector_to_array(self.domain_knn[key].hnsw.offsets)
                self.domain_nb[key] = faiss.vector_to_array(
                    self.domain_knn[key].hnsw.neighbors)
    def __del__(self):
        if self.flag:
            faiss.write_index(self.DB,self.index_path)
    def nb_filter(self,VT,nb_set):
        to_remove = {-1} | VT
        nb_set.difference_update(to_remove)
        return nb_set
    def dual_space_nb_filter(self,VT,nb_set,tag):
        if -1 in nb_set:
            nb_set.remove(-1)
        for item in nb_set.copy():
            if self.domain_id_dict[tag][item] in VT:
                nb_set.remove(item)
        return nb_set
    def Tag(self,batch_size):
        """

        :param batch_size: bs of matrix sent to vllm
        :return: None but save the tag pickle dumps
        """
        lm=summary_LLM(str(TAG_MODEL_PATH))

        messages=[[ {"role": "system", "content": EXTRACT_TAG_PROMPT},{"role": "user", "content": f'text document:{self.get_text_chunk([[i]])}'}] for i in range(self.DB.ntotal)]
        loop=len(messages)//batch_size
        overflow=len(messages)%batch_size
        for i in range(loop):
            ans=lm.generate(messages[i*batch_size:(i+1)*batch_size])
            split_ans=[x.split("</think>")[-1] for x in ans]
            print(f'loop:{i},bs:{batch_size}')
            self.tag.extend(split_ans)
            TAG_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(TAG_CACHE_PATH, "wb") as f:
                pickle.dump(self.tag, f)
                print('successfully save current pkl')
        if overflow!=0:
            ans = lm.generate(messages[loop * batch_size:self.DB.ntotal])
            print(f'addition loop')
            split_ans = [x.split("</think>")[-1] for x in ans]
            self.tag.extend(split_ans)
            TAG_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(TAG_CACHE_PATH, "wb") as f:
                pickle.dump(self.tag, f)
                print('successfully save current pkl')


    def normal_search_for_comparison(self,embedded_query,k):
        """function to process query from edge
           make sure that embedded_query is 2D np array of float32
           k: topK nearest
           return:
           1.entry point's distance
           2.entry point's id"""
        return self.DB.search(embedded_query,k)
    def search_by_edge_info(self,embedded_query,entry_info,k):
        """
        :param embedded_query: np array of 2d of embedded_query,should be of float32 and continuity
        :param entry_info: (dist,id) can be just the result of edge search
        :param k: topK search
        :return: dist,ids of result
        """
        D = np.zeros((len(embedded_query), k), dtype="float32")
        I = np.zeros((len(embedded_query), k), dtype="int64")
        self.DB.search_level_0(n=len(embedded_query),
                             x=faiss.swig_ptr(embedded_query),
                             k=k,
                             nearest=faiss.swig_ptr(entry_info[1].astype('int32')),
                             nearest_d=faiss.swig_ptr(entry_info[0]),
                             distances=faiss.swig_ptr(D),
                             labels=faiss.swig_ptr(I),
                             nprobe=1,
                             search_type=1, )
        return D,I
    '''
    n:number of query
    x:n*dim matrix of queries(you should transfer using swig_ptr)
    k:topk
    nearest: [nprobe] indices of nearest point ptr matrix
    nearest_d: [nprobe] np matrix(n*nprobe) corresponding to the nearest precomputed point distance ptr
    distances:receive distance ptr
    labels:receive result index ptr
    nprobe:number of entry point
    search_type:1 for one search step per entry point,2 for normal search
    '''
    def get_text_chunk(self,ids):
        """
        :param ids: array of retrieve ids
        :return:array of text chunks(not concat)
        """
        doc_ids = [self.index_to_docstore_id[i] for i in ids[0]]
        similar_docs = [self.docstore.search(doc_id) for doc_id in doc_ids]
        return similar_docs
    def set_search_params(self,ef_search):
        self.DB.hnsw.efSearch=ef_search

    def get_edge_cloud_pair(self,edge_entry_array,real_ids_array,save_path):
        """
        Build a mapping from edge entry IDs to ground-truth IDs.
        """
        pair=dict()
        for i in range(len(edge_entry_array)):
            if int(edge_entry_array[i][0]) not in pair.keys():
                pair[int(edge_entry_array[i][0])]=[]
            pair[int(edge_entry_array[i][0])].append(int(real_ids_array[i]))
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(pair, f, indent=4, ensure_ascii=False)
    def add_index(self,vectors):
        """
        :param vectors: np array of 2d of vectors, should be of float32

        :return: None
        """
        self.flag=True
        self.DB.add(vectors)
    def nn_assign(self,centroid_ids,mode):
        """
        Assign every data point to the nearest centroid.
        """
        xb = np.array([self.DB.reconstruct(i) for i in range(self.DB.ntotal)])
        np.save(str(OUTPUT_ROOT / 'xb.npy'), xb)
        centroid_points = xb[centroid_ids]
        if mode=='normal':
            temp_index=faiss.IndexFlatL2(768)
            temp_index.add(centroid_points)
            _,indices=temp_index.search(xb,1)
        elif mode=='anns':
            temp_index = faiss.IndexHNSWFlat(768, 64)
            efSearch = 256
            efConstruction = 256
            temp_index.hnsw.efSearch=efSearch
            temp_index.hnsw.efConstruction=efConstruction
            temp_index.add(centroid_points)
            _, indices = temp_index.search(xb, 1)
        result = [[] for _ in range(len(centroid_ids))]
        for data_idx, nearest_centroid_idx in enumerate(indices):
            result[nearest_centroid_idx[0]].append(data_idx)

        return result
    def adaptive_id_sample(self,update_packet,s1,s2,mode):

        real_centroid_id=update_packet.centroid_id
        centroid_ids=update_packet.sample_array[0]
        assign_num=update_packet.sample_array[1]
        temp_assign_ids=self.nn_assign(real_centroid_id,mode)
        assign_ids=[]
        sample_ids=[]
        for item in centroid_ids:
            assign_ids.append(temp_assign_ids[real_centroid_id.index(item)])
        for i in range(len(assign_ids)):
            if len(assign_ids[i])<assign_num[i]:
                sample_ids.extend(assign_ids[i])
            else:
                for j in range(len(assign_ids[i])):
                    if bool_generator(assign_num[i]/len(assign_ids[i])) and j!=centroid_ids[i]:
                        sample_ids.append(assign_ids[i][j])
                sample_ids.append(centroid_ids[i])
        sample_ids.extend(update_packet.no_update_array)
        xb = np.array([self.DB.reconstruct(i) for i in range(self.DB.ntotal)])
        xb=xb[sample_ids]
        out_dir = OUTPUT_ROOT / 'medcorp' / 'xb_id'
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(str(out_dir / f'{s1}_{s2}_50_50_xb.npy'), xb)
        np.save(str(out_dir / f'{s1}_{s2}_50_50_id.npy'), sample_ids)
        return sample_ids
    def walk_sampling(self,update_packet,prob_array,s1,s2):
        real_centroid_id = update_packet.centroid_id#all centroids before update decision
        centroid_ids = update_packet.sample_array[0]
        assign_num = update_packet.sample_array[1]
        Visit=set()
        Visit.update(update_packet.no_update_array)
        Visit.update(centroid_ids)
        sample_ids=[]
        for i in range(len(centroid_ids)):
            extra = 0
            tot_neighbor=[]
            tot_neighbor.append(centroid_ids[i])
            for prob in prob_array:
                init_neighbors = set()
                for n in tot_neighbor:

                    init_neighbors.update(self.neighbor[self.offset[n]:self.offset[n+1]])
                init_valid_neighbors = self.nb_filter(Visit,init_neighbors)
                if prob==0:
                    tot_neighbor=init_valid_neighbors
                    continue
                else:
                    number=math.ceil(assign_num[i]*prob)
                    if len(init_valid_neighbors)>(number+extra):
                        choice=random.sample(init_valid_neighbors,number)
                        sample_ids.extend(choice)
                        Visit.update(choice)
                        temp_tot_neighbor=[x for x in init_valid_neighbors if x not in choice]
                        if len(temp_tot_neighbor)>100:
                            tot_neighbor=random.sample(temp_tot_neighbor,100)
                        else:
                            tot_neighbor=temp_tot_neighbor
                    else:
                        extra=number+extra-len(init_valid_neighbors)
                        sample_ids.extend(init_valid_neighbors)
                        Visit.update(init_valid_neighbors)
                        choice=random.sample(init_valid_neighbors,math.floor(len(init_valid_neighbors)/2))
                        temp_tot_neighbor = [x for x in init_valid_neighbors if x not in choice]
                        if len(temp_tot_neighbor) > 100:
                            tot_neighbor = random.sample(temp_tot_neighbor, 100)
                        else:
                            tot_neighbor = temp_tot_neighbor
        sample_ids.extend(centroid_ids)
        sample_ids.extend(update_packet.no_update_array)
        xb=[]
        for item in sample_ids:
            xb.append( self.DB.reconstruct(int(item)))
        out_dir = OUTPUT_ROOT / 'medcorp' / 'xb_id'
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(str(out_dir / f'{s1}_{s2}_50_50_xb.npy'), xb)
        np.save(str(out_dir / f'{s1}_{s2}_50_50_id.npy'), sample_ids)
        return sample_ids
    def nav_walk_sampling(self,update_packet,prob_array,s1,s2):
        real_centroid_id = update_packet.centroid_id#all centroids before update decision
        centroid_ids = [x.id for x in update_packet.sample_array]

        Visit=set()
        Visit.update(update_packet.no_update_array)
        Visit.update(centroid_ids)
        sample_ids=[]
        for item in update_packet.sample_array:
            if item.node_property=='leaf':
                extra = 0
                tot_neighbor=[]
                tot_neighbor.append(item.id)
                for prob in prob_array:
                    init_neighbors = set()
                    for n in tot_neighbor:

                        init_neighbors.update(self.neighbor[self.offset[n]:self.offset[n+1]])
                    init_valid_neighbors = self.nb_filter(Visit,init_neighbors)
                    if prob==0:
                        tot_neighbor=init_valid_neighbors
                        continue
                    else:
                        number=math.ceil(item.assign_num*prob)
                        if len(init_valid_neighbors)>(number+extra):
                            choice=random.sample(init_valid_neighbors,number+extra)
                            sample_ids.extend(choice)
                            Visit.update(choice)
                            temp_tot_neighbor=[x for x in init_valid_neighbors if x not in choice]
                            if len(temp_tot_neighbor)>100:
                                tot_neighbor=random.sample(temp_tot_neighbor,100)
                            else:
                                tot_neighbor=temp_tot_neighbor
                        else:
                            extra=number+extra-len(init_valid_neighbors)
                            sample_ids.extend(init_valid_neighbors)
                            Visit.update(init_valid_neighbors)
                            choice=random.sample(init_valid_neighbors,math.floor(len(init_valid_neighbors)/2))
                            temp_tot_neighbor = [x for x in init_valid_neighbors if x not in choice]
                            if len(temp_tot_neighbor) > 100:
                                tot_neighbor = random.sample(temp_tot_neighbor, 100)
                            else:
                                tot_neighbor = temp_tot_neighbor
            else:#assert:item.node_property=='centroid'
                extra = 0
                tot_neighbor = []
                tot_neighbor.append(item.id)
                for prob in prob_array:
                    init_neighbors = set()
                    for n in tot_neighbor:
                        init_neighbors.update(self.neighbor[self.offset[n]:self.offset[n + 1]])
                    init_valid_neighbors = self.nb_filter(Visit, init_neighbors)
                    if prob == 0:
                        tot_neighbor = init_valid_neighbors
                        continue
                    else:
                        number = math.ceil(item.assign_num * prob)
                        if len(init_valid_neighbors) > (number + extra):
                            dist_list=[]
                            x0=np.array([self.DB.reconstruct(int(x)) for x in item.leaf])
                            for m in range(len(item.leaf)):
                                nb_vector=np.array([self.DB.reconstruct(int(x)) for x in init_valid_neighbors])
                                distances = np.linalg.norm(nb_vector - x0[m], axis=1)
                                id_distance_pairs = list(zip(init_valid_neighbors, distances))
                                sorted_pairs = sorted(id_distance_pairs, key=lambda x: x[1])
                                sorted_ids = [pair[0] for pair in sorted_pairs]
                                dist_list.append(sorted_ids)
                            nav_nb=merge_polling(dist_list)
                            choice = nav_nb[0:number+extra]
                            sample_ids.extend(choice)
                            Visit.update(choice)
                            temp_tot_neighbor = [x for x in init_valid_neighbors if x not in choice]
                            if len(temp_tot_neighbor) > 100:
                                tot_neighbor = random.sample(temp_tot_neighbor, 100)
                            else:
                                tot_neighbor = temp_tot_neighbor
                        else:
                            extra = number + extra - len(init_valid_neighbors)
                            sample_ids.extend(init_valid_neighbors)
                            Visit.update(init_valid_neighbors)
                            choice = random.sample(init_valid_neighbors, math.floor(len(init_valid_neighbors) / 2))
                            temp_tot_neighbor = [x for x in init_valid_neighbors if x not in choice]
                            if len(temp_tot_neighbor) > 100:
                                tot_neighbor = random.sample(temp_tot_neighbor, 100)
                            else:
                                tot_neighbor = temp_tot_neighbor
        sample_ids.extend(centroid_ids)
        sample_ids.extend(update_packet.no_update_array)
        xb=[]
        for item in sample_ids:
            xb.append( self.DB.reconstruct(int(item)))
        out_dir = OUTPUT_ROOT / 'ultra_vdb' / 'xb_id'
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(str(out_dir / f'{s1}_{s2}_50_50_xb.npy'), xb)
        np.save(str(out_dir / f'{s1}_{s2}_50_50_id.npy'), sample_ids)
        return sample_ids
    def sampling(self, update_packet, prob_array, id, output_dir=None):
        real_centroid_id = update_packet.centroid_id  # all centroids before update decision
        centroid_ids = [x.id for x in update_packet.sample_array]
        Visit = set()
        Visit.update(update_packet.no_update_array)  # do not forget to add no update in the end
        Visit.update(centroid_ids)
        sample_ids = []
        for item in update_packet.sample_array:
            if item.node_property == 'leaf':
                extra = 0
                tot_neighbor = []
                tot_neighbor.append(item.id)
                for prob in prob_array:
                    init_neighbors = set()
                    for n in tot_neighbor:
                        init_neighbors.update(self.neighbor[self.offset[n]:self.offset[n + 1]])
                    init_valid_neighbors = self.nb_filter(Visit, init_neighbors)
                    if prob == 0:
                        tot_neighbor = init_valid_neighbors
                        continue
                    else:
                        number = math.ceil(item.assign_num * prob)
                        if len(init_valid_neighbors) > (number + extra):
                            choice = random.sample(init_valid_neighbors, number + extra)
                            sample_ids.extend(choice)
                            Visit.update(choice)
                            temp_tot_neighbor = [x for x in init_valid_neighbors if x not in choice]
                            if len(temp_tot_neighbor) > 100:
                                tot_neighbor = random.sample(temp_tot_neighbor, 100)
                            else:
                                tot_neighbor = temp_tot_neighbor
                        else:
                            extra = number + extra - len(init_valid_neighbors)
                            sample_ids.extend(init_valid_neighbors)
                            Visit.update(init_valid_neighbors)
                            choice = random.sample(init_valid_neighbors, math.floor(len(init_valid_neighbors) / 2))
                            temp_tot_neighbor = [x for x in init_valid_neighbors if x not in choice]
                            if len(temp_tot_neighbor) > 100:
                                tot_neighbor = random.sample(temp_tot_neighbor, 100)
                            else:
                                tot_neighbor = temp_tot_neighbor
            elif item.node_property=='centroid':  # assert:item.node_property=='centroid'
                extra = 0
                tot_neighbor = []
                tot_neighbor.append(item.id)
                for prob in prob_array:
                    init_neighbors = set()
                    for n in tot_neighbor:
                        init_neighbors.update(self.neighbor[self.offset[n]:self.offset[n + 1]])
                    init_valid_neighbors = self.nb_filter(Visit, init_neighbors)
                    if prob == 0:
                        tot_neighbor = init_valid_neighbors
                        continue
                    else:
                        number = math.ceil(item.assign_num * prob)
                        if len(init_valid_neighbors) > (number + extra):
                            dist_list = []
                            x0 = np.array([self.DB.reconstruct(int(x)) for x in item.leaf])
                            for m in range(len(item.leaf)):
                                nb_vector = np.array([self.DB.reconstruct(int(x)) for x in init_valid_neighbors])
                                distances = np.linalg.norm(nb_vector - x0[m], axis=1)
                                id_distance_pairs = list(zip(init_valid_neighbors, distances))
                                sorted_pairs = sorted(id_distance_pairs, key=lambda x: x[1])
                                sorted_ids = [pair[0] for pair in sorted_pairs]
                                dist_list.append(sorted_ids)
                            nav_nb = merge_polling(dist_list)
                            choice = nav_nb[0:number + extra]
                            sample_ids.extend(choice)
                            Visit.update(choice)
                            temp_tot_neighbor = [x for x in init_valid_neighbors if x not in choice]
                            if len(temp_tot_neighbor) > 100:
                                tot_neighbor = random.sample(temp_tot_neighbor, 100)
                            else:
                                tot_neighbor = temp_tot_neighbor
                        else:
                            extra = number + extra - len(init_valid_neighbors)
                            sample_ids.extend(init_valid_neighbors)
                            Visit.update(init_valid_neighbors)
                            choice = random.sample(init_valid_neighbors, math.floor(len(init_valid_neighbors) / 2))
                            temp_tot_neighbor = [x for x in init_valid_neighbors if x not in choice]
                            if len(temp_tot_neighbor) > 100:
                                tot_neighbor = random.sample(temp_tot_neighbor, 100)
                            else:
                                tot_neighbor = temp_tot_neighbor
            else: #assert item.node_property=='interest'

                assert item.node_property == 'interest'
                extra = 0
                tot_neighbor = []
                tot_neighbor.append(self.domain_id_dict[item.tag].index(item.id))
                for prob in prob_array:
                    init_neighbors = set()
                    for n in tot_neighbor:
                        init_neighbors.update(self.domain_nb[item.tag][self.domain_offset[item.tag][n]:self.domain_offset[item.tag][n+1]])
                    init_valid_neighbors = self.dual_space_nb_filter(Visit,init_neighbors,item.tag)
                    if prob == 0:
                        tot_neighbor = init_valid_neighbors
                        continue
                    else:
                        number = math.ceil(item.assign_num * prob)
                        if len(init_valid_neighbors) > (number + extra):
                            dist_list = []
                            x0 = np.array([self.DB.reconstruct(int(x)) for x in item.leaf])
                            for m in range(len(item.leaf)):
                                nb_vector = np.array([self.domain_knn[item.tag].reconstruct(int(x)) for x in init_valid_neighbors])
                                distances = np.linalg.norm(nb_vector - x0[m], axis=1)
                                id_distance_pairs = list(zip(init_valid_neighbors, distances))
                                sorted_pairs = sorted(id_distance_pairs, key=lambda x: x[1])
                                sorted_ids = [pair[0] for pair in sorted_pairs]
                                dist_list.append(sorted_ids)
                            nav_nb = merge_polling(dist_list)
                            choice = nav_nb[0:number + extra]
                            real_choice=[self.domain_id_dict[item.tag][x] for x in choice]
                            sample_ids.extend(real_choice)
                            Visit.update(real_choice)
                            temp_tot_neighbor = [x for x in init_valid_neighbors if x not in choice]
                            if len(temp_tot_neighbor) > 100:
                                tot_neighbor = random.sample(temp_tot_neighbor, 100)
                            else:
                                tot_neighbor = temp_tot_neighbor
                        else:
                            extra = number + extra - len(init_valid_neighbors)
                            real_choice=[self.domain_id_dict[item.tag][x] for x in init_valid_neighbors]
                            sample_ids.extend(real_choice)
                            Visit.update(real_choice)
                            choice = random.sample(init_valid_neighbors, math.floor(len(init_valid_neighbors) / 2))
                            temp_tot_neighbor = [x for x in init_valid_neighbors if x not in choice]
                            if len(temp_tot_neighbor) > 100:
                                tot_neighbor = random.sample(temp_tot_neighbor, 100)
                            else:
                                tot_neighbor = temp_tot_neighbor

        sample_ids.extend(centroid_ids)
        sample_ids.extend(update_packet.no_update_array)
        xb = []
        for item in sample_ids:
            xb.append(self.DB.reconstruct(int(item)))

        if output_dir is None:
            out_dir = OUTPUT_ROOT / 'updates' / f'user_{id}'
        else:
            out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(str(out_dir / f'user_{id}_xb.npy'), np.asarray(xb, dtype='float32'))
        np.save(str(out_dir / f'user_{id}_id.npy'), np.asarray(sample_ids, dtype='int64'))
        with open(out_dir / f'user_{id}_update.pkl', 'wb') as f:
            pickle.dump((update_packet.sample_array, update_packet.no_update_array), f)
        return sample_ids







