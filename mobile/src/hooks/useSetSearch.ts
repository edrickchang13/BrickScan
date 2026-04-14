import { useInfiniteQuery, UseInfiniteQueryResult } from '@tanstack/react-query';
import { useRef, useEffect, useState } from 'react';
import { Config } from '../constants/config';

export interface SetSummary {
  setNum: string;
  setName: string;
  year: number;
  theme: string;
  partCount: number;
  imageUrl?: string;
}

interface SetSearchResponse {
  sets: SetSummary[];
  page: number;
  pageSize: number;
  total: number;
  hasMore: boolean;
}

async function searchSets(
  query: string,
  theme?: string,
  page: number = 1
): Promise<SetSearchResponse> {
  const params = new URLSearchParams({
    q: query,
    page: page.toString(),
    pageSize: Config.PAGE_SIZE.toString(),
    ...(theme && { theme }),
  });

  const response = await fetch(
    `${Config.API_BASE_URL}/api/sets/search?${params}`,
    {
      headers: { 'Content-Type': 'application/json' },
    }
  );

  if (!response.ok) {
    throw new Error(`Search failed: ${response.statusText}`);
  }

  return response.json();
}

interface UseSetSearchReturn
  extends UseInfiniteQueryResult<SetSearchResponse, Error> {
  debouncedSearch: (query: string, theme?: string) => void;
  isSearching: boolean;
}

export const useSetSearch = (): UseSetSearchReturn => {
  const [searchQuery, setSearchQuery] = useState('');
  const [searchTheme, setSearchTheme] = useState<string | undefined>();
  const debounceTimer = useRef<NodeJS.Timeout>();
  const [isSearching, setIsSearching] = useState(false);

  const query = useInfiniteQuery({
    queryKey: ['setSearch', searchQuery, searchTheme],
    queryFn: ({ pageParam = 1 }) =>
      searchSets(searchQuery, searchTheme, pageParam),
    getNextPageParam: (lastPage) =>
      lastPage.hasMore ? lastPage.page + 1 : undefined,
    initialPageParam: 1,
    enabled: searchQuery.length > 0,
    staleTime: 5 * 60 * 1000,
    gcTime: 30 * 60 * 1000,
    retry: 2,
  });

  const debouncedSearch = (query: string, theme?: string) => {
    setIsSearching(true);

    if (debounceTimer.current) {
      clearTimeout(debounceTimer.current);
    }

    debounceTimer.current = setTimeout(() => {
      setSearchQuery(query);
      setSearchTheme(theme);
      setIsSearching(false);
    }, 300);
  };

  useEffect(() => {
    return () => {
      if (debounceTimer.current) {
        clearTimeout(debounceTimer.current);
      }
    };
  }, []);

  return {
    ...query,
    debouncedSearch,
    isSearching,
  };
};
