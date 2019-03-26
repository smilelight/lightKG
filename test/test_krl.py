import sys
sys.path.append('/home/lightsmile/Projects/MyGithub/lightKG')

from lightkg.krl import KRL

train_path = '/home/lightsmile/NLP/corpus/kg/baike/train.sample.csv'
dev_path = '/home/lightsmile/NLP/corpus/kg/baike/test.sample.csv'
model_type = 'TransE'

krl = KRL()
# krl.train(train_path, model_type=model_type, dev_path=train_path, save_path='./krl_{}_saves'.format(model_type))

krl.load(save_path='./krl_{}_saves'.format(model_type), model_type=model_type)
# krl.test(train_path)

print(krl.predict_head(rel='外文名', tail='Compiler'))
print(krl.predict_rel(head='编译器', tail='Compiler'))
print(krl.predict_tail(head='编译器', rel='外文名'))
print(krl.predict(head='编译器', rel='外文名', tail='Compiler'))
