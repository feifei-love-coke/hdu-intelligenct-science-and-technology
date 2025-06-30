train_df = pd.DataFrame(train_data)

# 手动录入测试数据
test_data = {
    '色泽': ['青绿'],
    '根蒂': ['蜷缩'],
    '敲声': ['浊响'],
    '纹理': ['清晰'],
    '脐部': ['凹陷'],
    '触感': ['硬滑'],
    '密度': [0.697],
    '含糖量': [0.460]
}
test_df = pd.DataFrame(test_data)

# 训练模型并预测
X_train = train_df.drop('好瓜', axis=1)
y_train = train_df['好瓜']
model = NaiveBayes(X_train, y_train)

X_test = test_df
y_pre = model.predict(X_test)

print("预测结果:", y_pre)
print("预测结果解释: 1表示好瓜，0表示坏瓜")