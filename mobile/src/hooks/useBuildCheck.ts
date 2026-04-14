import { useQuery, UseQueryResult } from '@tanstack/react-query';
import { Config } from '../constants/config';

export interface MissingPart {
  partNum: string;
  colorHex: string;
  colorName: string;
  quantity: number;
}

export interface BuildCheckResult {
  setNum: string;
  setName: string;
  totalParts: number;
  totalQuantity: number;
  haveParts: number;
  haveQuantity: number;
  missingParts: MissingPart[];
  percentComplete: number;
}

async function fetchBuildCheck(setNum: string): Promise<BuildCheckResult> {
  const response = await fetch(
    `${Config.API_BASE_URL}/api/build-check/${setNum}`,
    {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
    }
  );

  if (!response.ok) {
    throw new Error(`Build check failed: ${response.statusText}`);
  }

  return response.json();
}

export const useBuildCheck = (
  setNum: string
): UseQueryResult<BuildCheckResult, Error> => {
  return useQuery({
    queryKey: ['buildCheck', setNum],
    queryFn: () => fetchBuildCheck(setNum),
    staleTime: 10 * 60 * 1000,
    gcTime: 60 * 60 * 1000,
    enabled: !!setNum,
    retry: 2,
    retryDelay: (attemptIndex) => Math.min(1000 * 2 ** attemptIndex, 30000),
  });
};
