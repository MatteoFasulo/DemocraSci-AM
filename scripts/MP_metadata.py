import pandas as pd
from neo4j import GraphDatabase

from const import AUTH, URI


def get_neo4j_driver(uri, auth):
    return GraphDatabase.driver(uri, auth=auth)


def get_mp_metadata(tx):
    query = """
    MATCH (p:`Unique Person`)-[:MEMBER_OF]->(party:Party)
    RETURN DISTINCT
        p.uid AS uid,
        p.first_name AS first_name,
        p.last_name AS last_name,
        p.first_name + " " + p.last_name AS fullname,
        p.gender AS gender,
        party.abbrev AS party,
        party.name AS party_name
    ORDER BY last_name ASC
    """
    result = tx.run(query)
    return pd.DataFrame(dict(record) for record in result)


driver = get_neo4j_driver(URI, AUTH)
with driver.session() as session:
    df_party_map = session.execute_read(get_mp_metadata)

driver.close()
df_party_map.to_csv("mp_metadata.csv", index=False)
