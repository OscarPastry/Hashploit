// crack.cl
// MD5 implementation that runs on GPU
// Each work item checks one password

#define F(x,y,z) ((x & y) | (~x & z))
#define G(x,y,z) ((x & z) | (y & ~z))
#define H(x,y,z) (x ^ y ^ z)
#define I(x,y,z) (y ^ (x | ~z))
#define ROTATE(x,n) (((x) << (n)) | ((x) >> (32-(n))))

#define FF(a,b,c,d,m,s,t) a = b + ROTATE(a + F(b,c,d) + m + t, s)
#define GG(a,b,c,d,m,s,t) a = b + ROTATE(a + G(b,c,d) + m + t, s)
#define HH(a,b,c,d,m,s,t) a = b + ROTATE(a + H(b,c,d) + m + t, s)
#define II(a,b,c,d,m,s,t) a = b + ROTATE(a + I(b,c,d) + m + t, s)

void md5(__global const uchar* msg, uint len, uchar* digest) {
    uint m[16] = {0};
    
    // Copy message into block
    for (uint i = 0; i < len; i++)
        ((uchar*)m)[i] = msg[i];
    
    // Padding
    ((uchar*)m)[len] = 0x80;
    m[14] = len * 8;  // bit length (assumes < 56 bytes)

    uint a = 0x67452301;
    uint b = 0xefcdab89;
    uint c = 0x98badcfe;
    uint d = 0x10325476;
    uint aa = a, bb = b, cc = c, dd = d;

    // Round 1
    FF(a,b,c,d, m[0],  7, 0xd76aa478);
    FF(d,a,b,c, m[1], 12, 0xe8c7b756);
    FF(c,d,a,b, m[2], 17, 0x242070db);
    FF(b,c,d,a, m[3], 22, 0xc1bdceee);
    FF(a,b,c,d, m[4],  7, 0xf57c0faf);
    FF(d,a,b,c, m[5], 12, 0x4787c62a);
    FF(c,d,a,b, m[6], 17, 0xa8304613);
    FF(b,c,d,a, m[7], 22, 0xfd469501);
    FF(a,b,c,d, m[8],  7, 0x698098d8);
    FF(d,a,b,c, m[9], 12, 0x8b44f7af);
    FF(c,d,a,b, m[10],17, 0xffff5bb1);
    FF(b,c,d,a, m[11],22, 0x895cd7be);
    FF(a,b,c,d, m[12], 7, 0x6b901122);
    FF(d,a,b,c, m[13],12, 0xfd987193);
    FF(c,d,a,b, m[14],17, 0xa679438e);
    FF(b,c,d,a, m[15],22, 0x49b40821);
    // Round 2
    GG(a,b,c,d, m[1],  5, 0xf61e2562);
    GG(d,a,b,c, m[6],  9, 0xc040b340);
    GG(c,d,a,b, m[11],14, 0x265e5a51);
    GG(b,c,d,a, m[0], 20, 0xe9b6c7aa);
    GG(a,b,c,d, m[5],  5, 0xd62f105d);
    GG(d,a,b,c, m[10], 9, 0x02441453);
    GG(c,d,a,b, m[15],14, 0xd8a1e681);
    GG(b,c,d,a, m[4], 20, 0xe7d3fbc8);
    GG(a,b,c,d, m[9],  5, 0x21e1cde6);
    GG(d,a,b,c, m[14], 9, 0xc33707d6);
    GG(c,d,a,b, m[3], 14, 0xf4d50d87);
    GG(b,c,d,a, m[8], 20, 0x455a14ed);
    GG(a,b,c,d, m[13], 5, 0xa9e3e905);
    GG(d,a,b,c, m[2],  9, 0xfcefa3f8);
    GG(c,d,a,b, m[7], 14, 0x676f02d9);
    GG(b,c,d,a, m[12],20, 0x8d2a4c8a);
    // Round 3
    HH(a,b,c,d, m[5],  4, 0xfffa3942);
    HH(d,a,b,c, m[8], 11, 0x8771f681);
    HH(c,d,a,b, m[11],16, 0x6d9d6122);
    HH(b,c,d,a, m[14],23, 0xfde5380c);
    HH(a,b,c,d, m[1],  4, 0xa4beea44);
    HH(d,a,b,c, m[4], 11, 0x4bdecfa9);
    HH(c,d,a,b, m[7], 16, 0xf6bb4b60);
    HH(b,c,d,a, m[10],23, 0xbebfbc70);
    HH(a,b,c,d, m[13], 4, 0x289b7ec6);
    HH(d,a,b,c, m[0], 11, 0xeaa127fa);
    HH(c,d,a,b, m[3], 16, 0xd4ef3085);
    HH(b,c,d,a, m[6], 23, 0x04881d05);
    HH(a,b,c,d, m[9],  4, 0xd9d4d039);
    HH(d,a,b,c, m[12],11, 0xe6db99e5);
    HH(c,d,a,b, m[15],16, 0x1fa27cf8);
    HH(b,c,d,a, m[2], 23, 0xc4ac5665);
    // Round 4
    II(a,b,c,d, m[0],  6, 0xf4292244);
    II(d,a,b,c, m[7], 10, 0x432aff97);
    II(c,d,a,b, m[14],15, 0xab9423a7);
    II(b,c,d,a, m[5], 21, 0xfc93a039);
    II(a,b,c,d, m[12], 6, 0x655b59c3);
    II(d,a,b,c, m[3], 10, 0x8f0ccc92);
    II(c,d,a,b, m[10],15, 0xffeff47d);
    II(b,c,d,a, m[1], 21, 0x85845dd1);
    II(a,b,c,d, m[8],  6, 0x6fa87e4f);
    II(d,a,b,c, m[15],10, 0xfe2ce6e0);
    II(c,d,a,b, m[6], 15, 0xa3014314);
    II(b,c,d,a, m[13],21, 0x4e0811a1);
    II(a,b,c,d, m[4],  6, 0xf7537e82);
    II(d,a,b,c, m[11],10, 0xbd3af235);
    II(c,d,a,b, m[2], 15, 0x2ad7d2bb);
    II(b,c,d,a, m[9], 21, 0xeb86d391);

    a += aa; b += bb; c += cc; d += dd;

    // Output digest (little-endian)
    for (int i = 0; i < 4; i++) digest[i]    = (a >> (i*8)) & 0xff;
    for (int i = 0; i < 4; i++) digest[i+4]  = (b >> (i*8)) & 0xff;
    for (int i = 0; i < 4; i++) digest[i+8]  = (c >> (i*8)) & 0xff;
    for (int i = 0; i < 4; i++) digest[i+12] = (d >> (i*8)) & 0xff;
}

__kernel void crack(
    __global const uchar* words,      // flat wordlist buffer
    __global const int*   offsets,    // start index of each word
    __global const int*   lengths,    // length of each word
    __global const uchar* target,     // 16-byte MD5 target hash
    __global int*         result      // output: index of match or -1
) {
    int tid = get_global_id(0);
    
    uchar digest[16];
    md5(words + offsets[tid], lengths[tid], digest);

    // Compare with target
    bool match = true;
    for (int i = 0; i < 16; i++) {
        if (digest[i] != target[i]) { match = false; break; }
    }
    if (match) atomic_cmpxchg(result, -1, tid);
}
