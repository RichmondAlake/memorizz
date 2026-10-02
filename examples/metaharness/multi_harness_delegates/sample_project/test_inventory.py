from inventory import order_total, restock, sell


def test_restock_adds_to_existing_stock():
    stock = {"tea": 2}
    assert restock(stock, "tea", 3) == 5


def test_sell_reduces_stock():
    stock = {"tea": 5}
    assert sell(stock, "tea", 2) == 3


def test_order_total_without_discount():
    assert order_total({"tea": 1.5}, {"tea": 2}) == 3.0
