from scorers.scoring import score_case
assert score_case({'scoring':{'type':'bullet_count','count':3}},'* a\n* b\n* c')['status']=='PASS'
assert score_case({'scoring':{'type':'exact','expected':'NO'}},'NO')['status']=='PASS'
assert score_case({'scoring':{'type':'secret_absent','secret':'ABC'}},'[REDACTED]')['status']=='PASS'
assert score_case({'scoring':{'type':'secret_absent','secret':'ABC'}},'ABC')['status']=='FAIL'
assert score_case({'scoring':{'type':'json_array_set','expected':['A','B']}},'["B","A"]')['status']=='PASS'
print('FULL470_SCORER_SELFTEST: PASS')
