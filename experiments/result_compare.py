import json
import sys



def compare(attack_name='untargeted_EMITIDIAIFGSM_BayesOpt',blackbox_method='untargeted_EMITIDIAIFGSM_BayesOpt'):


    f = open('log/imagenet/'+attack_name+'/'+attack_name+'.json')
    data = json.load(f)


    labels_attacked = []

    hybrid_result = data["result"]
    #print(hybrid_result)
    hybrid_reuslt_overall = hybrid_result["Final Result"]
    #print(hybrid_reuslt_overall)
    del hybrid_result["Final Result"]
    query_count = 0
    for label in hybrid_result:
        if hybrid_result[label]["transfer_flag"]==0 and hybrid_result[label]["attack_flag"]==1:
            #print(hybrid_result[label]["query"])
            query_count += hybrid_result[label]["query"]
            labels_attacked.append(label)
    if hybrid_reuslt_overall["success"] != 0:
        avg = int(hybrid_reuslt_overall["success"])
    else:
        avg = int(1)
    query_avg_hybrid= query_count/avg
    print("query_avg_hybrid:", query_avg_hybrid)

    f = open('log/imagenet/'+blackbox_method+'/'+blackbox_method+'.json')
    data = json.load(f)

    blackbox_result = data["result"]
    query_count_blackbox = 0
    for label in labels_attacked:
        if label in blackbox_result.keys():
            query_count_blackbox+=blackbox_result[label]["query"]

    query_avg_blackbox = query_count_blackbox/avg


    print("query_avg_blackbox:",query_avg_blackbox)



def count(attack_name='untargeted_EMITIDIAIFGSM_BayesOpt'):
    f = open('log/imagenet/'+attack_name+'/'+attack_name+'.json')
    data = json.load(f)
    result = data["result"]
    del result["Final Result"]
    
    query_count = 0
    attack_succ=0
    for label in result:
        query_count += result[label]["query"]
        if result[label]["attack_flag"] ==1:
            attack_succ+=1

    query_avg= float(query_count/len(result))
    print("query_avg:", query_avg)
    print("Succ:", attack_succ)

if __name__ == "__main__":
    args = sys.argv
    if len(args)==3:

        compare(args[1],args[2])
    else:
        count(args[1])


