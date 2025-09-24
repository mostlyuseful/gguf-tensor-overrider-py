from functools import cached_property
import io
import requests
from requests.exceptions import RequestException
import logging

# Constants
DEFAULT_BLOCK_SIZE = 1024 # bytes
RANGE_HEADER_PROBE = "bytes=0-0"
CONTENT_LENGTH_HEADER = "Content-Length"
CONTENT_RANGE_HEADER = "Content-Range"
RANGE_HEADER = "Range"


# Custom exceptions
class HttpFileError(Exception):
    """Base exception for HttpFile operations."""
    pass


class FileLengthError(HttpFileError):
    """Exception raised when file length cannot be determined."""
    pass


class DataFetchError(HttpFileError):
    """Exception raised when data cannot be fetched from the remote file."""
    pass

class InMemoryCache:
    """A simple cache strategy that caches everything in memory."""
    
    def __init__(self) -> None:
        self.cache: dict[tuple[int, int], bytes] = {}
        logging.debug("InMemoryCache initialized.")

    def get(self, range: tuple[int, int]) -> bytes | None:
        """Retrieve cached data for the given byte range.
        
        Args:
            range: Tuple of (start_byte, end_byte) defining the range
            
        Returns:
            Cached bytes data if found, None otherwise
        """
        value = self.cache.get(range, None)
        return value

    def set(self, range: tuple[int, int], data: bytes) -> None:
        """Store data in cache for the given byte range.
        
        Args:
            range: Tuple of (start_byte, end_byte) defining the range
            data: Bytes data to cache
        """
        self.cache[range] = data
        logging.debug(f"InMemoryCache.set({range}, <{len(data)} bytes>, <{sum(len(v) for v in self.cache.values())/1024**2:.1f} MB total size>)")

class HttpFile:
    """A file-like object that fetches data from a URL in blocks and caches it.
    
    This class provides a file-like interface for reading data from HTTP URLs,
    implementing block-based caching to optimize performance for repeated reads.
    """
    
    def __init__(self, 
                 url: str, 
                 block_size: int = DEFAULT_BLOCK_SIZE, 
                 cache_strategy: InMemoryCache | None = None, 
                 extra_headers: dict[str, str] | None = None, 
                 session: requests.Session | None = None) -> None:
        """Initialize the HttpFile.
        
        Args:
            url: The URL to fetch data from
            block_size: Size of blocks for caching (default: 1024 bytes)
            cache_strategy: Cache implementation to use (default: InMemoryCache)
            extra_headers: Additional HTTP headers to send with requests
            session: Requests session to use (default: creates new session)
        """
        self.url = url
        self.block_size = block_size
        self.cache = cache_strategy if cache_strategy else InMemoryCache()
        self.extra_headers = extra_headers or {}
        self.session = session if session else requests.Session()
        self.offset = 0
        self.content = None
        logging.debug(f"HttpFile initialized: url={url}, block_size={block_size}, extra_headers={self.extra_headers}")

    def _get_length_from_head(self) -> int | None:
        """Try to get file length using HEAD request.
        
        Returns:
            File length in bytes if successful, None if Content-Length not available
            
        Raises:
            FileLengthError: If the HTTP request fails
        """
        logging.debug(f"HttpFile._get_length_from_head: Sending HEAD request to {self.url}")
        try:
            response = self.session.head(self.url, headers=self.extra_headers, allow_redirects=True)
            response.raise_for_status()
            length = response.headers.get(CONTENT_LENGTH_HEADER)
            if length is not None and int(length) > 0:
                logging.debug(f"HttpFile._get_length_from_head: Content-Length from HEAD={length}")
                return int(length)
            return None
        except RequestException as e:
            logging.debug(f"HttpFile._get_length_from_head: HEAD request failed: {e}")
            raise FileLengthError(f"Failed to get file length: {e}")

    def _parse_content_range(self, content_range: str) -> int | None:
        """Parse the Content-Range header to extract total file size.
        
        Args:
            content_range: Content-Range header value (e.g., "bytes 0-0/123456")
            
        Returns:
            Total file size in bytes if parsing successful, None otherwise
        """
        try:
            # Content-Range: bytes 0-0/123456
            total = content_range.split('/')[-1]
            logging.debug(f"HttpFile._parse_content_range: Parsed total from Content-Range: {total}")
            return int(total)
        except (ValueError, IndexError):
            logging.debug(f"HttpFile._parse_content_range: Failed to parse Content-Range: {content_range}")
            return None

    def _get_length_from_range_request(self) -> int | None:
        """Try to get file length using a range request.
        
        Returns:
            File length in bytes if successful, None if not available
            
        Raises:
            FileLengthError: If the HTTP request fails
        """
        logging.debug("HttpFile._get_length_from_range_request: Trying GET with Range: bytes=0-0")
        headers = self.extra_headers.copy()
        headers[RANGE_HEADER] = RANGE_HEADER_PROBE
        
        try:
            get_response = self.session.get(self.url, headers=headers, allow_redirects=True)
            get_response.raise_for_status()
            
            # Try Content-Range first
            content_range = get_response.headers.get(CONTENT_RANGE_HEADER)
            if content_range:
                total = self._parse_content_range(content_range)
                if total is not None:
                    return total
            
            # Fallback to Content-Length
            length = get_response.headers.get(CONTENT_LENGTH_HEADER)
            if length is not None and int(length) > 0:
                logging.debug(f"HttpFile._get_length_from_range_request: Content-Length from GET={length}")
                return int(length)
                
            return None
        except RequestException as e:
            logging.debug(f"HttpFile._get_length_from_range_request: GET request failed: {e}")
            raise FileLengthError(f"Failed to get file length: {e}")

    @cached_property
    def file_length(self) -> int:
        """Get the total length of the file, handling redirects and missing Content-Length."""
        # First try HEAD request
        length = self._get_length_from_head()
        if length is not None:
            return length
        
        # If HEAD fails, try range request
        length = self._get_length_from_range_request()
        if length is not None:
            return length
        
        logging.debug("HttpFile.file_length: Could not determine file length from HEAD or GET")
        raise FileLengthError("Content-Length header is missing and could not be determined via GET")

    def _calculate_block_range(self, current_offset: int) -> tuple[int, int]:
        """Calculate the block range for a given offset.
        
        Args:
            current_offset: The current byte offset in the file
            
        Returns:
            Tuple of (block_start, block_end) defining the block boundaries
        """
        block_start = (current_offset // self.block_size) * self.block_size
        block_end = min(block_start + self.block_size, self.file_length)
        return (block_start, block_end)

    def _get_block_data(self, block_range: tuple[int, int]) -> bytes:
        """Fetch block data from cache or HTTP request.
        
        Args:
            block_range: Tuple of (start_byte, end_byte) defining the block range
            
        Returns:
            Block data as bytes
            
        Raises:
            DataFetchError: If the HTTP request fails
        """
        block_start, block_end = block_range
        
        # Check cache first
        cached_data = self.cache.get(block_range)
        if cached_data is not None:
            return cached_data
        
        # Cache miss - fetch from HTTP
        logging.debug(f"HttpFile._get_block_data: Cache MISS for block_range={block_range}, sending HTTP GET")
        headers = self.extra_headers.copy()
        headers[RANGE_HEADER] = f'bytes={block_start}-{block_end - 1}'
        
        try:
            response = self.session.get(self.url, headers=headers)
            response.raise_for_status()
            block_data = response.content
            self.cache.set(block_range, block_data)
            logging.debug(f"HttpFile._get_block_data: Received {len(block_data)} bytes for block_range={block_range}")
            return block_data
        except RequestException as e:
            logging.debug(f"HttpFile._get_block_data: HTTP GET failed for block_range={block_range}: {e}")
            raise DataFetchError(f"Failed to fetch data: {e}")

    def _extract_relevant_data(self, block_data: bytes, block_start: int, current_offset: int, end_offset: int) -> bytes:
        """Extract the relevant portion of block data based on the requested range.
        
        Args:
            block_data: The complete block data
            block_start: Starting byte offset of the block
            current_offset: Current reading position
            end_offset: End position for reading
            
        Returns:
            Subset of block data relevant to the requested range
        """
        start_in_block = max(0, current_offset - block_start)
        end_in_block = min(len(block_data), end_offset - block_start)
        return block_data[start_in_block:end_in_block]

    def read(self, size: int = -1) -> bytes:
        """Read the specified number of bytes from the file.
        
        Args:
            size: Number of bytes to read. If -1 or negative, reads until EOF.
                 If 0, returns empty bytes.
                 
        Returns:
            Bytes data read from the file
            
        Raises:
            DataFetchError: If fetching data from the remote file fails
        """
        
        if size == 0:
            logging.debug("HttpFile.read: size=0, returning empty bytes")
            return b""
        
        if size < 0:
            size = self.file_length - self.offset
            logging.debug(f"HttpFile.read: size<0, using size={size}")
        
        if self.offset >= self.file_length:
            logging.debug("HttpFile.read: offset >= file_length, returning empty bytes")
            return b""
        
        start_offset_bytes = self.offset
        end_offset_bytes = min(self.offset + size, self.file_length)
        data = bytearray()
        current_offset_bytes = start_offset_bytes
        
        while current_offset_bytes < end_offset_bytes:
            block_range = self._calculate_block_range(current_offset_bytes)
            block_start, _ = block_range
            
            block_data = self._get_block_data(block_range)
            
            relevant_data = self._extract_relevant_data(
                block_data, block_start, current_offset_bytes, end_offset_bytes
            )
            data.extend(relevant_data)
            current_offset_bytes += len(relevant_data)

        # update offset
        self.offset = end_offset_bytes
        
        return data

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> None:
        """Seek to a specified offset in the file.
        
        Args:
            offset: Byte offset to seek to
            whence: How to interpret the offset:
                   - io.SEEK_SET (0): absolute position
                   - io.SEEK_CUR (1): relative to current position 
                   - io.SEEK_END (2): relative to end of file
        """
        old_offset = self.offset
        if whence == io.SEEK_CUR:
            self.offset += offset
        elif whence == io.SEEK_END:
            self.offset = self.file_length + offset
        else:  # io.SEEK_SET
            self.offset = offset
        logging.debug(f"HttpFile.seek(offset={offset}, whence={whence}): {old_offset} -> {self.offset}")

    def close(self) -> None:
        """Close the HttpFile and clean up resources."""
        logging.debug("HttpFile.close() called.")

    def __enter__(self) -> 'HttpFile':
        """Enter the context manager."""
        logging.debug("HttpFile.__enter__() called.")
        return self

    def __exit__(self, exc_type: type | None, exc_value: Exception | None, traceback: object | None) -> None:
        """Exit the context manager."""
        logging.debug(f"HttpFile.__exit__() called. exc_type={exc_type}, exc_value={exc_value}")
        self.close()

# Usage example
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, force=True)

    url ="https://huggingface.co/ubergarm/Qwen3-235B-A22B-Thinking-2507-GGUF/resolve/main/IQ4_KSS/Qwen3-235B-A22B-Thinking-2507-IQ4_KSS-00001-of-00003.gguf"
    with HttpFile(url, block_size=DEFAULT_BLOCK_SIZE, cache_strategy=InMemoryCache(), extra_headers={}) as f:
        data: bytes = f.read(8)  # Sends the GET request for a full block and returns the first 8 bytes
        print(data)
        
        f.seek(7 * 1024**2)  # Seek to a specific position
        data = f.read(2 * 1024)  # Read more data
        print(data[:16])  # Print the first 16 bytes of the read data
