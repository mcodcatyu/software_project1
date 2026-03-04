import ll as ll # modify programe name to test different programes
def test_main():
    energy_val, order_val =ll.main('ll', 50, 50, 0.5, 0)
    assert energy_val<0 and 0<=order_val<=1

def test_main2():
    energy_val, order_val = ll.main('ll', 1000, 20, 0.65, 0)
    assert energy_val<-1100 and 0.6<=order_val<=0.9

