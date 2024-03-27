def keyword_search(query, client, properties=["text", "title", "url", "views", "lang"], num_results=3):
    try:
        where_filter = {
            "path": ["lang"],
            "operator": "Equal",
            "valueString": "en"
        }

        response = (
            client.query.get("Articles", properties)
            .with_bm25(query=query)
            .with_where(where_filter)
            .with_limit(num_results)
            .do()
        )

        return response.get('data', {}).get('Get', {}).get('Articles', [])
    except Exception as e:
        print(f"Error in keyword_search: {e}")
        return []




def dense_retrieval(query, 
                    client,
                    results_lang='en', 
                    properties=["text", "title", "url", "views", "lang"],
                    num_results=5):
    nearText = {"concepts": [query]}
    
    # To filter by language
    where_filter = {
        "path": ["lang"],
        "operator": "Equal",
        "valueString": results_lang
    }

    response = (
        client.query.get("Articles", properties)
            .with_near_text(nearText)
            .with_where(where_filter)
            .with_limit(num_results)
            .with_additional("distance")
            .do()
    )

    return response



def print_result(result):
    """ Print results with colorful formatting """
    for i, item in enumerate(result):
        # Check if the item is a dictionary before attempting to access its keys
        if isinstance(item, dict):
            print(f'Item {i}:')
            for key, value in item.items():
                print(f"{key}: {value}\n")
        else:
            print(f'Item {i}: {item}\n')
        print()  # Extra newline for better separation

