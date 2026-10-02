# Corner shop inventory

`inventory.py` keeps stock counts for a small shop.

- `restock(stock, item, quantity)` adds stock. Quantities must be positive.
- `sell(stock, item, quantity)` removes stock and refuses to oversell. Selling an
  item the shop has never stocked raises `ValueError`.
- `low_stock(stock, threshold=5)` lists items at or below the threshold.
- `order_total(prices, basket, discount=0)` prices a basket; `discount` is a
  percentage, so `discount=10` takes 10% off.

Run the tests with `pytest -q`.
