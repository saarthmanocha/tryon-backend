"""
TryOn Backend - Product Search
Find where to buy garments using reverse image search.

Uses SerpAPI Google Lens for image-based product search.
Includes filtering for Indian e-commerce stores and price comparison.
"""

import os
import re
from dataclasses import dataclass, field
from typing import Optional, List
from datetime import datetime

# SerpAPI client (pip install google-search-results)
try:
    from serpapi import GoogleSearch
    SERPAPI_AVAILABLE = True
except ImportError:
    SERPAPI_AVAILABLE = False
    print("Warning: serpapi not installed. Run: pip install google-search-results")


@dataclass
class ProductResult:
    """A single product result from search."""
    title: str
    price: Optional[str] = None
    price_value: Optional[float] = None  # Numeric price for sorting
    currency: str = "INR"
    store: str = "Unknown"
    link: str = ""
    thumbnail: Optional[str] = None
    rating: Optional[float] = None
    reviews_count: Optional[int] = None
    in_stock: bool = True
    
    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "price": self.price,
            "price_value": self.price_value,
            "currency": self.currency,
            "store": self.store,
            "link": self.link,
            "thumbnail": self.thumbnail,
            "rating": self.rating,
            "reviews_count": self.reviews_count,
            "in_stock": self.in_stock
        }


@dataclass
class ProductSearchResult:
    """Result of product search."""
    success: bool
    query_image: Optional[str] = None
    products: List[ProductResult] = field(default_factory=list)
    total_found: int = 0
    search_time_ms: Optional[int] = None
    searched_at: Optional[str] = None
    error: Optional[str] = None
    
    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "query_image": self.query_image,
            "products": [p.to_dict() for p in self.products],
            "total_found": self.total_found,
            "search_time_ms": self.search_time_ms,
            "searched_at": self.searched_at,
            "error": self.error
        }


class ProductSearcher:
    """
    Search for products using reverse image search.
    
    Uses SerpAPI Google Lens to find similar products across
    e-commerce platforms with prices and purchase links.
    """
    
    # Preferred Indian e-commerce stores (prioritized in results)
    INDIAN_STORES = [
        "amazon.in", "flipkart.com", "myntra.com", "ajio.com",
        "nykaa.com", "tatacliq.com", "snapdeal.com", "meesho.com",
        "shopclues.com", "limeroad.com", "jabong.com", "koovs.com",
        "bewakoof.com", "souledstore.com", "fynd.com", "pernia.com"
    ]
    
    # International stores (included but lower priority)
    GLOBAL_STORES = [
        "amazon.com", "ebay.com", "asos.com", "zara.com", "hm.com",
        "uniqlo.com", "shein.com", "aliexpress.com"
    ]
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize with SerpAPI key."""
        self.api_key = api_key or os.getenv("SERPAPI_KEY")
        
        if not self.api_key:
            print("Warning: SERPAPI_KEY not set. Product search will not work.")
    
    def search_by_image_url(
        self,
        image_url: str,
        max_results: int = 10,
        country: str = "in",  # Default to India
        include_global: bool = True
    ) -> ProductSearchResult:
        """
        Search for products matching the given image.
        
        Args:
            image_url: URL of the garment image (e.g., Cloudinary URL)
            max_results: Maximum number of results to return
            country: Country code for localized results (default: India)
            include_global: Whether to include international stores
            
        Returns:
            ProductSearchResult with matched products
        """
        if not SERPAPI_AVAILABLE:
            return ProductSearchResult(
                success=False,
                error="SerpAPI not installed. Run: pip install google-search-results"
            )
        
        if not self.api_key:
            return ProductSearchResult(
                success=False,
                error="SERPAPI_KEY not configured"
            )
        
        try:
            start_time = datetime.now()
            
            # Configure Google Lens search
            params = {
                "engine": "google_lens",
                "url": image_url,
                "api_key": self.api_key,
                "hl": "en",  # Language
                "country": country,
            }
            
            search = GoogleSearch(params)
            results = search.get_dict()
            
            # Extract visual matches (shopping results)
            visual_matches = results.get("visual_matches", [])
            
            # Process and filter results
            products = []
            for match in visual_matches:
                product = self._parse_product(match)
                if product:
                    products.append(product)
            
            # Sort and filter
            products = self._sort_and_filter(
                products, 
                max_results, 
                include_global
            )
            
            end_time = datetime.now()
            search_time = int((end_time - start_time).total_seconds() * 1000)
            
            return ProductSearchResult(
                success=True,
                query_image=image_url,
                products=products,
                total_found=len(products),
                search_time_ms=search_time,
                searched_at=datetime.now().isoformat()
            )
            
        except Exception as e:
            return ProductSearchResult(
                success=False,
                query_image=image_url,
                error=str(e)
            )
    
    def _parse_product(self, match: dict) -> Optional[ProductResult]:
        """Parse a visual match into a ProductResult."""
        try:
            title = match.get("title", "")
            if not title:
                return None
            
            # Extract source/store info
            source = match.get("source", "")
            link = match.get("link", "")
            thumbnail = match.get("thumbnail", "")
            
            # Extract price if available
            price_info = match.get("price", {})
            if isinstance(price_info, dict):
                price_str = price_info.get("value", "")
                currency = price_info.get("currency", "INR")
            elif isinstance(price_info, str):
                price_str = price_info
                currency = "INR"
            else:
                price_str = ""
                currency = "INR"
            
            # Parse numeric price
            price_value = self._parse_price(price_str)
            
            # Extract rating if available
            rating = match.get("rating")
            if rating:
                try:
                    rating = float(rating)
                except:
                    rating = None
            
            reviews = match.get("reviews")
            if reviews:
                try:
                    reviews = int(reviews)
                except:
                    reviews = None
            
            # Determine store name from source or link
            store = self._extract_store_name(source, link)
            
            return ProductResult(
                title=title,
                price=price_str if price_str else None,
                price_value=price_value,
                currency=currency,
                store=store,
                link=link,
                thumbnail=thumbnail,
                rating=rating,
                reviews_count=reviews
            )
            
        except Exception as e:
            print(f"Error parsing product: {e}")
            return None
    
    def _parse_price(self, price_str: str) -> Optional[float]:
        """Extract numeric price from string."""
        if not price_str:
            return None
        
        try:
            # Remove currency symbols and commas
            cleaned = re.sub(r'[₹$,\s]', '', str(price_str))
            # Extract number
            match = re.search(r'[\d.]+', cleaned)
            if match:
                return float(match.group())
        except:
            pass
        
        return None
    
    def _extract_store_name(self, source: str, link: str) -> str:
        """Extract clean store name from source or URL."""
        if source:
            return source
        
        if link:
            try:
                from urllib.parse import urlparse
                domain = urlparse(link).netloc
                # Remove www. prefix
                domain = domain.replace("www.", "")
                # Get first part before TLD
                store = domain.split(".")[0]
                return store.capitalize()
            except:
                pass
        
        return "Unknown"
    
    def _sort_and_filter(
        self, 
        products: List[ProductResult],
        max_results: int,
        include_global: bool
    ) -> List[ProductResult]:
        """
        Sort products by relevance and price.
        Prioritize Indian stores.
        """
        if not products:
            return []
        
        # Separate Indian and global products
        indian_products = []
        global_products = []
        
        for product in products:
            is_indian = any(
                store in product.link.lower() 
                for store in self.INDIAN_STORES
            )
            
            if is_indian:
                indian_products.append(product)
            elif include_global:
                global_products.append(product)
        
        # Sort each group by price (lowest first)
        def sort_key(p):
            return p.price_value if p.price_value else float('inf')
        
        indian_products.sort(key=sort_key)
        global_products.sort(key=sort_key)
        
        # Combine: Indian first, then global
        all_products = indian_products + global_products
        
        return all_products[:max_results]


# === API Functions ===

def search_products(
    image_url: str,
    max_results: int = 10,
    country: str = "in",
    include_global: bool = True
) -> dict:
    """
    Search for products matching the given image.
    
    Args:
        image_url: URL of the garment image
        max_results: Maximum results to return (default 10)
        country: Country code (default "in" for India)
        include_global: Include international stores
        
    Returns:
        Dictionary with product results
    """
    searcher = ProductSearcher()
    result = searcher.search_by_image_url(
        image_url=image_url,
        max_results=max_results,
        country=country,
        include_global=include_global
    )
    return result.to_dict()


# === CLI ===

if __name__ == "__main__":
    import sys
    import json
    
    if len(sys.argv) < 2:
        print("Usage: python product_search.py <image_url>")
        print("Make sure SERPAPI_KEY environment variable is set")
        sys.exit(1)
    
    image_url = sys.argv[1]
    result = search_products(image_url)
    print(json.dumps(result, indent=2))
