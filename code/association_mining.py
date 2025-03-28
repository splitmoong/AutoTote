import pandas as pd
#for fp growth (frequent item set generation)
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth

dataset = pd.read_csv("../datasets/groceries_dataset.csv")


def user_transactions(user):

    user = user
    df = dataset
    df = df.groupby(["Member_number", "Date"])["itemDescription"].apply(list).reset_index()
    df.rename(columns={"itemDescription": "Items"}, inplace=True)

    if user != "all":
        df = df[df["Member_number"] == user]

    df = df.drop(columns=["Member_number", "Date"]).reset_index(drop=True)
    return df


#one time function to convert dataset
# def augment_dataset():
#     df = dataset
#     df = df.groupby(["Member_number", "Date"], as_index=False)["itemDescription"].apply(list)
#     df.rename(columns={"itemDescription": "Items"}, inplace=True)
#     print(df.head())
#     df.to_csv("../datasets/exports/transactions.csv", index=False)

#generate frequent itemsets

def generate_frequent_itemsets(temp, support):
    #to convert the transactions into a one hot pandas matrix of items x transactions
    te = TransactionEncoder()
    temp = [transaction[0] if isinstance(transaction[0], list) else transaction for transaction in temp]
    transactions_matrix = te.fit(temp).transform(temp)
    transactions_df = pd.DataFrame(transactions_matrix, columns=te.columns_)

    #now running fp growth on the matrix
    freq_itemset = fpgrowth(transactions_df, min_support=support, use_colnames=True)
    print(freq_itemset)


transactions = user_transactions(user="all")
generate_frequent_itemsets(transactions.values.tolist(), support=0.1)
