import {
  useMutation,
  useQuery,
  useInfiniteQuery,
  MutationOptions,
} from '@tanstack/react-query';
import { Share } from 'react-native';
import * as FileSystem from 'expo-file-system/legacy';
import { Config } from '../constants/config';
import { inventoryToCsv } from '../utils/csvUtils';

export interface InventoryItem {
  id: string;
  partNum: string;
  partName: string;
  colorHex: string;
  colorName: string;
  quantity: number;
  imageUrl?: string;
  createdAt: string;
  updatedAt: string;
}

interface InventoryListParams {
  search?: string;
  colorFilter?: string;
  page: number;
  pageSize?: number;
}

interface InventoryListResponse {
  items: InventoryItem[];
  total: number;
  page: number;
  pageSize: number;
  hasMore: boolean;
}

async function fetchInventoryList(
  params: InventoryListParams
): Promise<InventoryListResponse> {
  const queryParams = new URLSearchParams({
    page: params.page.toString(),
    pageSize: (params.pageSize || Config.PAGE_SIZE).toString(),
    ...(params.search && { search: params.search }),
    ...(params.colorFilter && { colorFilter: params.colorFilter }),
  });

  const response = await fetch(
    `${Config.API_BASE_URL}/api/inventory?${queryParams}`,
    {
      headers: { 'Content-Type': 'application/json' },
    }
  );

  if (!response.ok) {
    throw new Error(`Failed to fetch inventory: ${response.statusText}`);
  }

  return response.json();
}

async function addToInventory(item: Omit<InventoryItem, 'id' | 'createdAt' | 'updatedAt'>): Promise<InventoryItem> {
  const response = await fetch(`${Config.API_BASE_URL}/api/inventory`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(item),
  });

  if (!response.ok) {
    throw new Error(`Failed to add item: ${response.statusText}`);
  }

  return response.json();
}

async function updateQuantity(
  id: string,
  quantity: number
): Promise<InventoryItem> {
  const response = await fetch(`${Config.API_BASE_URL}/api/inventory/${id}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ quantity }),
  });

  if (!response.ok) {
    throw new Error(`Failed to update quantity: ${response.statusText}`);
  }

  return response.json();
}

async function deleteItem(id: string): Promise<void> {
  const response = await fetch(`${Config.API_BASE_URL}/api/inventory/${id}`, {
    method: 'DELETE',
  });

  if (!response.ok) {
    throw new Error(`Failed to delete item: ${response.statusText}`);
  }
}

async function exportInventoryAsCSV(items: InventoryItem[]): Promise<void> {
  const csvContent = inventoryToCsv(items);
  const filename = `inventory-${new Date().toISOString().split('T')[0]}.csv`;
  const documentDirectory = FileSystem.documentDirectory;
  if (!documentDirectory) {
    throw new Error('Document directory is unavailable on this device');
  }
  const filePath = `${documentDirectory}${filename}`;

  await FileSystem.writeAsStringAsync(filePath, csvContent);

  await Share.share({
    url: filePath,
    title: 'Export Inventory',
    message: `Exported ${items.length} inventory items`,
  });
}

export const useInventoryList = (
  search?: string,
  colorFilter?: string
) => {
  return useInfiniteQuery({
    queryKey: ['inventory', { search, colorFilter }],
    queryFn: ({ pageParam = 1 }) =>
      fetchInventoryList({
        search,
        colorFilter,
        page: pageParam,
        pageSize: Config.PAGE_SIZE,
      }),
    getNextPageParam: (lastPage) =>
      lastPage.hasMore ? lastPage.page + 1 : undefined,
    initialPageParam: 1,
  });
};

export const useAddToInventory = (
  options?: MutationOptions<
    InventoryItem,
    Error,
    Omit<InventoryItem, 'id' | 'createdAt' | 'updatedAt'>
  >
) => {
  return useMutation({
    mutationFn: addToInventory,
    ...options,
  });
};

export const useUpdateQuantity = (
  options?: MutationOptions<InventoryItem, Error, { id: string; quantity: number }>
) => {
  return useMutation({
    mutationFn: ({ id, quantity }) => updateQuantity(id, quantity),
    ...options,
  });
};

export const useDeleteItem = (
  options?: MutationOptions<void, Error, string>
) => {
  return useMutation({
    mutationFn: deleteItem,
    ...options,
  });
};

export const useExportCSV = () => {
  return useMutation({
    mutationFn: exportInventoryAsCSV,
  });
};
