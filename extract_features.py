import re
from urllib.parse import urlparse
import tldextract
import socket
from functools import lru_cache
from collections import OrderedDict
import asyncio
import aiohttp

# Precompile regex patterns for efficiency
IP_PATTERN = re.compile(r'^\d{1,3}(?:\.\d{1,3}){3}$')
WORD_PATTERN = re.compile(r'[A-Za-z0-9]+')
SUBDOMAIN_PATTERN = re.compile(r'^w+\d*$')

# Cache DNS lookups to avoid redundant network calls
@lru_cache(maxsize=10000)
def cached_gethostbyname(hostname):
    try:
        socket.gethostbyname(hostname)
        return 1
    except Exception:
        return 0

# Define the feature order used during training
FEATURE_ORDER = [
    'f1_url_length', 'f2_hostname_length', 'f3_has_ip', 'f4_dot', 'f5_hyphen',
    'f6_at', 'f7_question', 'f8_ampersand', 'f9_pipe', 'f10_equal',
    'f11_underscore', 'f12_tilde', 'f13_percent', 'f14_slash', 'f15_asterisk',
    'f16_colon', 'f17_comma', 'f18_semicolon', 'f19_dollar', 'f20_space_or_%20',
    'f21_www_count', 'f22_dotcom_count', 'f23_http_count', 'f24_double_slash_count',
    'f25_https', 'f26_digit_ratio_url', 'f27_digit_ratio_hostname', 'f28_punycode',
    'f29_port', 'f30_tld_in_path', 'f31_tld_in_subdomain', 'f32_abnormal_subdomain',
    'f33_subdomain_count', 'f34_prefix_suffix', 'f35_random_domain', 'f36_shortening_service',
    'f37_suspicious_extension', 'f38_redirection_count', 'f39_external_redirections',
    'f40_word_count_url', 'f41_max_char_repeat', 'f42_shortest_word_length_url',
    'f43_word_count_hostname', 'f44_word_count_path', 'f45_longest_word_length_url',
    'f46_longest_word_length_hostname', 'f47_longest_word_length_path', 'f48_avg_word_length_url',
    'f49_avg_word_length_hostname', 'f50_avg_word_length_path', 'f51_phish_hints',
    'f52_brand_in_domain', 'f53_brand_in_subdomain', 'f54_brand_in_path', 'f55_dns_record',
    'f56_suspicious_tld', 'f57_qty_dot_domain', 'f58_qty_hyphen_domain', 'f59_qty_underscore_domain',
    'f60_qty_at_domain', 'f61_qty_percent_domain', 'f62_qty_dot_path', 'f63_qty_hyphen_path',
    'f64_qty_slash_path', 'f65_qty_question_path', 'f66_qty_equal_path', 'f67_qty_dot_query',
    'f68_qty_hyphen_query', 'f69_qty_equal_query', 'f70_qty_ampersand_query', 'f71_qty_percent_query',
    'f72_length_domain', 'f73_length_path', 'f74_length_query', 'f75_number_of_directories',
    'f76_number_of_query_params', 'f77_presence_of_fragment', 'f78_number_of_encoded_chars',
    'f79_presence_of_email', 'f80_digit_ratio_domain', 'f81_special_char_ratio_path',
    'f82_is_encoded', 'f83_server_client_domain', 'f84_tld_length'
]

def extract_url_features(url):
    features = {}
    parsed = urlparse(url)
    hostname = parsed.netloc
    path = parsed.path
    query = parsed.query
    fragment = parsed.fragment
    ext = tldextract.extract(url)
    subdomain = ext.subdomain
    tld = ext.suffix
    domain_main = ext.domain
    domain_full = parsed.netloc
    url_lower = url.lower()

    # Original 56 Features (f1 to f56)
    features['f1_url_length'] = len(url)
    features['f2_hostname_length'] = len(hostname)
    features['f3_has_ip'] = 1 if IP_PATTERN.match(hostname) else 0
    special_chars = {
        'f4_dot': '.', 'f5_hyphen': '-', 'f6_at': '@', 'f7_question': '?',
        'f8_ampersand': '&', 'f9_pipe': '|', 'f10_equal': '=', 'f11_underscore': '_',
        'f12_tilde': '~', 'f13_percent': '%', 'f14_slash': '/', 'f15_asterisk': '*',
        'f16_colon': ':', 'f17_comma': ',', 'f18_semicolon': ';', 'f19_dollar': '$',
        'f20_space_or_%20': ['%20', ' ']
    }
    for key, char in special_chars.items():
        if key == 'f20_space_or_%20':
            features[key] = sum(url.count(c) for c in char)
        else:
            features[key] = url.count(char)
    features['f21_www_count'] = url_lower.count("www")
    features['f22_dotcom_count'] = url_lower.count(".com")
    features['f23_http_count'] = url_lower.count("http")
    features['f24_double_slash_count'] = url.count("//")
    features['f25_https'] = 1 if parsed.scheme.lower() == 'https' else 0
    digits_url = sum(c.isdigit() for c in url)
    features['f26_digit_ratio_url'] = digits_url / len(url) if len(url) > 0 else 0
    digits_hostname = sum(c.isdigit() for c in hostname)
    features['f27_digit_ratio_hostname'] = digits_hostname / len(hostname) if len(hostname) > 0 else 0
    features['f28_punycode'] = 1 if "xn--" in hostname else 0
    features['f29_port'] = 1 if parsed.port else 0
    features['f30_tld_in_path'] = 1 if tld and tld in path else 0
    features['f31_tld_in_subdomain'] = 1 if tld and tld in subdomain else 0
    features['f32_abnormal_subdomain'] = 1 if subdomain and subdomain.lower() != "www" and SUBDOMAIN_PATTERN.match(subdomain) else 0
    features['f33_subdomain_count'] = len(subdomain.split('.')) if subdomain else 0
    features['f34_prefix_suffix'] = 1 if '-' in domain_main else 0
    vowels = sum(1 for c in domain_main.lower() if c in 'aeiou')
    features['f35_random_domain'] = 1 if domain_main and (vowels / len(domain_main)) < 0.3 else 0
    shortening_services = {'bit.ly', 'tinyurl.com', 'goo.gl', 'ow.ly', 't.co', 'is.gd', 'buff.ly', 'adf.ly'}
    features['f36_shortening_service'] = 1 if hostname.lower() in shortening_services else 0
    suspicious_exts = {'.exe', '.js', '.txt'}
    features['f37_suspicious_extension'] = 1 if any(path.lower().endswith(ext) for ext in suspicious_exts) else 0

    # For f38 and f39, which were provided asynchronously during training, set default values
    features['f38_redirection_count'] = -1
    features['f39_external_redirections'] = -1

    # Continue with remaining features
    words_url = WORD_PATTERN.findall(url)
    features['f40_word_count_url'] = len(words_url)
    max_repeat = 0
    current_char = ''
    current_count = 0
    for char in url:
        if char == current_char:
            current_count += 1
        else:
            current_char = char
            current_count = 1
        max_repeat = max(max_repeat, current_count)
    features['f41_max_char_repeat'] = max_repeat
    features['f42_shortest_word_length_url'] = min((len(w) for w in words_url), default=0)
    words_hostname = WORD_PATTERN.findall(hostname)
    features['f43_word_count_hostname'] = len(words_hostname)
    words_path = WORD_PATTERN.findall(path)
    features['f44_word_count_path'] = len(words_path)
    features['f45_longest_word_length_url'] = max((len(w) for w in words_url), default=0)
    features['f46_longest_word_length_hostname'] = max((len(w) for w in words_hostname), default=0)
    features['f47_longest_word_length_path'] = max((len(w) for w in words_path), default=0)
    features['f48_avg_word_length_url'] = sum(len(w) for w in words_url) / len(words_url) if words_url else 0
    features['f49_avg_word_length_hostname'] = sum(len(w) for w in words_hostname) / len(words_hostname) if words_hostname else 0
    features['f50_avg_word_length_path'] = sum(len(w) for w in words_path) / len(words_path) if words_path else 0
    sensitive_words = {"login", "signin", "verify", "account", "update", "secure", "confirm", "bank", "paypal", "ebay", "admin", "security", "password"}
    features['f51_phish_hints'] = sum(url_lower.count(word) for word in sensitive_words)
    brands = {"google", "facebook", "amazon", "paypal", "apple", "microsoft", "ebay"}
    domain_lower = domain_main.lower()
    features['f52_brand_in_domain'] = 1 if any(brand in domain_lower for brand in brands) else 0
    features['f53_brand_in_subdomain'] = 1 if any(brand in subdomain.lower() for brand in brands) else 0
    features['f54_brand_in_path'] = 1 if any(brand in path.lower() for brand in brands) else 0

    suspicious_tlds = {"tk", "ml", "ga", "cf", "gq"}
    features['f56_suspicious_tld'] = 1 if tld.lower() in suspicious_tlds else 0

    # Additional 28 Features (f57 to f84)
    features['f57_qty_dot_domain'] = domain_full.count('.')
    features['f58_qty_hyphen_domain'] = domain_full.count('-')
    features['f59_qty_underscore_domain'] = domain_full.count('_')
    features['f60_qty_at_domain'] = domain_full.count('@')
    features['f61_qty_percent_domain'] = domain_full.count('%')
    features['f62_qty_dot_path'] = path.count('.')
    features['f63_qty_hyphen_path'] = path.count('-')
    features['f64_qty_slash_path'] = path.count('/')
    features['f65_qty_question_path'] = path.count('?')
    features['f66_qty_equal_path'] = path.count('=')
    features['f67_qty_dot_query'] = query.count('.')
    features['f68_qty_hyphen_query'] = query.count('-')
    features['f69_qty_equal_query'] = query.count('=')
    features['f70_qty_ampersand_query'] = query.count('&')
    features['f71_qty_percent_query'] = query.count('%')
    features['f72_length_domain'] = len(domain_full)
    features['f73_length_path'] = len(path)
    features['f74_length_query'] = len(query)
    features['f75_number_of_directories'] = len(path.split('/')) - 1 if path and path != '/' else 0
    features['f76_number_of_query_params'] = len(query.split('&')) if query else 0
    features['f77_presence_of_fragment'] = 1 if fragment else 0
    features['f78_number_of_encoded_chars'] = url.count('%')
    features['f79_presence_of_email'] = 1 if 'mailto:' in url_lower else 0
    features['f80_digit_ratio_domain'] = sum(c.isdigit() for c in domain_full) / len(domain_full) if domain_full else 0
    features['f81_special_char_ratio_path'] = sum(not c.isalnum() for c in path) / len(path) if path else 0
    features['f82_is_encoded'] = 1 if '%' in url else 0
    features['f83_server_client_domain'] = 1 if any(word in domain_lower for word in {'server', 'client'}) else 0
    features['f84_tld_length'] = len(tld)

    # Return the features in the exact training order as an OrderedDict.
    ordered_features = OrderedDict((key, features.get(key, -1)) for key in FEATURE_ORDER)
    return ordered_features

# Asynchronous functions for batch processing remain unchanged
async def fetch_redirects(session, url):
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=5), allow_redirects=True) as response:
            history = response.history
            redirect_count = len(history)
            original_hostname = urlparse(url).netloc.lower()
            external_redirects = sum(1 for resp in history if urlparse(resp.url).netloc.lower() != original_hostname)
            return redirect_count, external_redirects
    except:
        return -1, -1

async def extract_all_features(session, url):
    # Use the synchronous extractor and then update with asynchronous results.
    features = extract_url_features(url)
    redirection_count, external_redirects = await fetch_redirects(session, url)
    features['f38_redirection_count'] = redirection_count
    features['f39_external_redirections'] = external_redirects
    # Update DNS record info
    parsed = urlparse(url)
    hostname = parsed.netloc
    features['f55_dns_record'] = cached_gethostbyname(hostname)
    # Reorder one more time to ensure consistency
    ordered_features = OrderedDict((key, features.get(key, -1)) for key in FEATURE_ORDER)
    return ordered_features

# (Optional) Additional async batch processing functions can remain here.
async def process_urls(urls):
    async with aiohttp.ClientSession() as session:
        tasks = [extract_all_features(session, url) for url in urls]
        return await asyncio.gather(*tasks)
