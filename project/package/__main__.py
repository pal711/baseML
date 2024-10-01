from package.models.dummy import add_model

def main():
    ip = list(range(10))
    ret = add_model(ip)
    print(ret)

if __name__ == "__main__":
    main()
