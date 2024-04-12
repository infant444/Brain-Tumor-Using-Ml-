export const BASE_URL="http://localhost:5000"
// import { environment } from "../../environment/environment.pord";

// const BASE_URL=environment.production?'':'http://localhost:5000';

export const BRAIN_TUMOR=BASE_URL+"/api/braintumor/test";
export const SAMPLE_PDF=BASE_URL+"/api/braintumor/report/";
export const LOGIN=BASE_URL+"/api/user/login";
export const SIGNUP=BASE_URL+"/api/user/signup";
export const LOGIN_GOOGLE=BASE_URL+"/api/user/login-google";
export const FEEDBACK=BASE_URL+"/api/user/feedback";
export const SENDOTP=BASE_URL+"/api/user/send-otp/";
export const CHANGEPASSWORD=BASE_URL+"/api/user/changepassword/";

export const GETCHAT=BASE_URL+"/api/chat/getchat/";
export const POSTCHAT=BASE_URL+"/api/chat/postchat";
export const GETREPORT=BASE_URL+"/api/braintumor/getreport/";
export const GETSIGLEREPORT=BASE_URL+"/api/braintumor/getsinglereport/";
export const GETAllUSER=BASE_URL+"/api/admin/alluser";
export const GETUSER=BASE_URL+"/api/admin/getuser/";
export const DELETEUSER=BASE_URL+"/api/admin/delete/user/";
export const GETALLMESSAGE=BASE_URL+"/api/admin/getallmessage";
export const SENDMESSAGE=BASE_URL+"/api/admin/sendmessage/";



