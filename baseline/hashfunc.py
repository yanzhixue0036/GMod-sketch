import hashlib
import struct

def sha1_hash32(data, seed=0):
    """A 32-bit hash function based on SHA1.

    Args:
        data (bytes or tuple): the data to generate 32-bit integer hash from.

    Returns:
        int: an integer hash value that can be encoded using 32 bits.
    """
    # 如果输入是元组，则将其转换为字节
    if isinstance(data, tuple):
        # 将元组转换为字符串，然后编码为字节
        data = str(data).encode('utf-8')
    
    return struct.unpack('<I', hashlib.sha1(data).digest()[:4])[0]

def sha1_hash64(data, seed=0):
    """A 64-bit hash function based on SHA1.

    Args:
        data (bytes): the data to generate 64-bit integer hash from.

    Returns:
        int: an integer hash value that can be encoded using 64 bits.
    """
    # 如果输入是元组，则将其转换为字节
    if isinstance(data, tuple):
        # 将元组转换为字符串，然后编码为字节
        data = str(data).encode('utf-8')
    return struct.unpack('<Q', hashlib.sha1(data).digest()[:8])[0]

import mmh3

def mmh3_hash32(data, seed):
    """A 32-bit hash function based on mmh3.

    Args:
        data (bytes or tuple): the data to generate 32-bit integer hash from.

    Returns:
        int: an integer hash value that can be encoded using 32 bits.
    """
    if isinstance(data, tuple):
        # Convert the tuple to a string, then encode it to bytes
        data = str(data).encode('utf-8')
    elif isinstance(data, str):
        # Encode the string to bytes directly
        data = data.encode('utf-8')
    
    # Use mmh3 to generate a 32-bit hash
    return mmh3.hash(data, seed=seed, signed=False)


def mmh3_hash64(data, seed):
    """A 64-bit hash function based on mmh3.

    Args:
        data (bytes or tuple): the data to generate 64-bit integer hash from.

    Returns:
        tuple: a tuple of two integers, representing the 64-bit hash.
    """
    if isinstance(data, tuple):
        # Convert the tuple to a string, then encode it to bytes
        data = str(data).encode('utf-8')
    elif isinstance(data, str):
        # Encode the string to bytes directly
        data = data.encode('utf-8')

    # Use mmh3 to generate a 64-bit hash (returns two 32-bit values)
    return mmh3.hash64(data, seed=seed, signed=False)
