import pandas as pd

dataset = pd.read_csv("../datasets/groceries_dataset.csv")


def user_transactions(user: int):
    df = dataset
    df = df.groupby(["Member_number", "Date"])["itemDescription"].apply(list).reset_index()
    df.rename(columns={"itemDescription": "Items"}, inplace=True)
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
def generate_frequent_itemsets():
    pass


print(user_transactions(1111))
