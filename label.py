import pickle

def main():
    data1 = {'suggestFix':0,
         'no': 1}
    output = open('label.pkl', 'wb')
    # Pickle dictionary using protocol 0.
    pickle.dump(data1, output)
    output.close()

if __name__ == '__main__':
    main()
