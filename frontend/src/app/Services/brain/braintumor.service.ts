import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { Observable, tap } from 'rxjs';
import {  Tumor_Info } from '../../model/brain';
import { BRAIN_TUMOR, GETREPORT, GETSIGLEREPORT, SAMPLE_PDF } from '../../model/url';
import { Patient } from '../../model/patient';

@Injectable({
  providedIn: 'root'
})
export class BraintumorService {

  constructor(private http:HttpClient) { }

  testbrain(file:any):Observable<Patient>{
    return this.http.post<Patient>(BRAIN_TUMOR,file)
  }

  // getreport(id:string){
  //   return this.http.get(SAMPLE_PDF+id, { responseType: 'arraybuffer' }).pipe(
  //     tap({
  //       next:(x)=>{
  //         var file = new Blob([x], { type: 'application/pdf' });
  //         var fileURL = URL.createObjectURL(file);
  //         // console.log(fileURL)
  //         window.open(fileURL);
  //       }
  //     })
  //   )}

    getreport(id:string){
    return this.http.get(SAMPLE_PDF+id, {responseType: 'blob' })}
  // pipe(
  //     tap({
  //       next:(x)=>{
  //         var file = new Blob([x], { type: 'application/pdf' });
  //         var fileURL = URL.createObjectURL(file);
  //         // console.log(fileURL)
  //         window.open(fileURL);
  //       }
  //     })
  //   )}
  getreports(id:string):Observable<Patient[]>{
    return this.http.get<Patient[]>(GETREPORT+id);
  }
  getsinglereports(id:string):Observable<Patient>{
    return this.http.get<Patient>(GETSIGLEREPORT+id);

  }
}
