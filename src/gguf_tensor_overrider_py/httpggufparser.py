from urllib.parse import urlparse
from pathlib import Path
import struct

from gguf_parser import GGUFParser, GGUFParseError
from gguf_tensor_overrider_py.httpfile import HttpFile


class HttpGGUFParser(GGUFParser):
    """A GGUF parser that supports both local files and HTTP(S) URLs."""

    def __init__(self, file_path_or_url):
        """Initialize the parser with the given GGUF file path or URL."""
        super().__init__(file_path_or_url)
        self.file_path_or_url = file_path_or_url

    def parse(self):
        """Parse the GGUF file from either a local path or HTTP(S) URL."""
        # Determine if we're dealing with a URL or local file
        parsed = urlparse(str(self.file_path_or_url))
        is_url = parsed.scheme in ("http", "https")

        if is_url:
            # Use HttpFile for URLs
            with HttpFile(str(self.file_path_or_url), block_size=5 * 1024**2) as f:
                self._parse_from_file_object(f)
        else:
            # Use regular file for local paths
            file_path = Path(self.file_path_or_url)
            if not file_path.exists():
                raise GGUFParseError(f"GGUF file not found: {file_path}")

            with open(file_path, "rb") as f:
                self._parse_from_file_object(f)

    def _parse_from_file_object(self, f):
        """Parse GGUF from a file-like object (adapted from original GGUFParser.parse)."""
        # Read the magic number
        self.magic_number = f.read(4)
        if self.magic_number != self.GGUF_MAGIC_NUMBER:
            raise GGUFParseError("Invalid magic number")

        # Read the version
        self.version = struct.unpack("I", f.read(4))[0]
        if self.version != 3:
            raise GGUFParseError("Unsupported version")

        # Read the number of tensors and metadata key-value pairs
        tensor_count = struct.unpack("Q", f.read(8))[0]
        metadata_kv_count = struct.unpack("Q", f.read(8))[0]

        # Read the metadata key-value pairs
        self.metadata = {}
        for _ in range(metadata_kv_count):
            key, value = self._read_metadata_kv(f)
            self.metadata[key] = value

        # Read the general.alignment metadata field
        self.alignment = self.metadata.get("general.alignment", 1)

        # Read the tensor information
        self.tensors_info = []
        for _ in range(tensor_count):
            tensor_info = self._read_tensor_info(f)
            self.tensors_info.append(tensor_info)

    def _read_string(self, f):
        """Read a string from the file."""
        length = struct.unpack("Q", f.read(8))[0]
        return f.read(length).decode("utf-8")

    def _read_metadata_kv(self, f):
        """Read a metadata key-value pair from the file."""
        key = self._read_string(f)
        value_type = struct.unpack("I", f.read(4))[0]
        value = self._read_value(f, value_type)
        return key, value

    def _read_value(self, f, value_type):
        """Read a value of the given type from the file."""
        if value_type in self.VALUE_FORMATS:
            return struct.unpack(
                self.VALUE_FORMATS[value_type],
                f.read(struct.calcsize(self.VALUE_FORMATS[value_type])),
            )[0]
        if value_type == 8:  # STRING
            return self._read_string(f)
        if value_type == 9:  # ARRAY
            array_type = struct.unpack("I", f.read(4))[0]
            array_len = struct.unpack("Q", f.read(8))[0]
            return [self._read_value(f, array_type) for _ in range(array_len)]
        raise GGUFParseError("Unsupported value type")

    def _read_tensor_info(self, f):
        """Read tensor information from the file."""
        name = self._read_string(f)
        n_dimensions = struct.unpack("I", f.read(4))[0]
        dimensions = struct.unpack(f"{n_dimensions}Q", f.read(8 * n_dimensions))
        tensor_type = struct.unpack("I", f.read(4))[0]
        offset = struct.unpack("Q", f.read(8))[0]
        return {
            "name": name,
            "n_dimensions": n_dimensions,
            "dimensions": dimensions,
            "type": tensor_type,
            "offset": offset,
        }
