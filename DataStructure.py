class query_packet:
    def __init__(self,text_query,embedded_query,entry_info):
        """Container for a query and its retrieved entry-point information.

        Args:
            text_query: Raw text query.
            embedded_query: Embedded query vector.
            entry_info: Tuple of (distance, id) returned by the edge search.
        """
        self.text_query=text_query
        self.embedded_query=embedded_query
        self.entry_info=entry_info
class update_packet:
    def __init__(self,centroid_id,sample_array,no_update_array):
        """Container for device-index update instructions."""
        self.centroid_id=centroid_id
        self.sample_array=sample_array
        self.no_update_array=no_update_array
class node_info:
    def __init__(self,node_id,node_property,assign_num,leaf=None,tag=None):
        """Describe a sampled node and its update budget."""
        self.id=node_id
        self.node_property=node_property
        self.assign_num=assign_num
        self.leaf=leaf
        self.tag=tag