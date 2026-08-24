Cache identity is the normalized query plus exact user_id, domain, and
data_version. Query normalization trims surrounding whitespace and casefolds
text; tenant, domain, and version values remain exact opaque strings. At
now >= expires_at a read is a miss and removes that entry.
invalidate(domain=..., data_version=...) removes only entries matching every
supplied filter; omitted filters are wildcards. __len__ reports the number of
stored live entries.
