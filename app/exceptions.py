class RAGError(Exception):
    """Base exception for expected application errors."""


class UnsupportedDocumentError(RAGError):
    pass


class EmptyDocumentError(RAGError):
    pass


class IndexNotReadyError(RAGError):
    pass


class ProviderError(RAGError):
    pass

