def received_request(req_type, endpoint, nb_items, size):
    print("="*size)
    print("Received {} request at {} with {} items".format(req_type, endpoint, nb_items).center(size, "-"))
    print("="*size)
